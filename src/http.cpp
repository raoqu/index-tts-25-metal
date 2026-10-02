#include "itts25/http.hpp"
#include "httplib.h"
#include <sqlite3.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <random>
#include <csignal>
#include <atomic>
#include <set>
namespace itts25 {
namespace {
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
struct APIError:std::runtime_error {int status;APIError(int s,const std::string& m):std::runtime_error(m),status(s) {}};
std::string read_file(const std::string& p){std::ifstream in(p,std::ios::binary);if(!in)throw APIError(400,"File not found: "+p);return std::string(std::istreambuf_iterator<char>(in),{});}
void write_file(const std::string& p,const std::string& data){std::ofstream out(p,std::ios::binary);out.write(data.data(),data.size());if(!out)throw std::runtime_error("Cannot write upload");}
std::string identity(const std::string& prefix){std::random_device r;std::ostringstream s;s<<prefix<<'_'<<std::hex;for(int i=0;i<4;i++)s<<r();return s.str();}
std::string now(){return std::to_string(std::time(nullptr));}
std::string absolute(std::string p){if(p.rfind("~/",0)==0){const char* home=std::getenv("HOME");if(home)p=std::string(home)+p.substr(1);}return fs::absolute(p).lexically_normal().string();}
bool as_bool(const Json& v,bool def=false){if(v.is_null())return def;if(v.is_boolean())return v.get<bool>();if(v.is_string()){auto s=unicode_transform(v.get<std::string>(),1);if(s=="true"||s=="1"||s=="yes"||s=="on")return true;if(s=="false"||s=="0"||s=="no"||s=="off"||s.empty())return false;}throw APIError(400,"Invalid boolean");}
struct TempDir {fs::path path=fs::temp_directory_path()/identity("itts25-native");TempDir(){fs::create_directories(path);}~TempDir(){std::error_code e;fs::remove_all(path,e);}};
class Store {
    sqlite3* db_=nullptr;std::mutex mutex_;std::string root_;
    void execute(const std::string& sql){char* error=nullptr;if(sqlite3_exec(db_,sql.c_str(),nullptr,nullptr,&error)!=SQLITE_OK){std::string e=error?error:"SQLite error";sqlite3_free(error);throw std::runtime_error(e);}}
    std::vector<Json> query(const std::string& sql,const std::vector<Json>& args={}){
        sqlite3_stmt* raw=nullptr;if(sqlite3_prepare_v2(db_,sql.c_str(),-1,&raw,nullptr)!=SQLITE_OK)throw std::runtime_error(sqlite3_errmsg(db_));std::unique_ptr<sqlite3_stmt,decltype(&sqlite3_finalize)> stmt(raw,sqlite3_finalize);
        for(size_t i=0;i<args.size();i++){auto& v=args[i];if(v.is_null())sqlite3_bind_null(raw,i+1);else if(v.is_boolean())sqlite3_bind_int(raw,i+1,v.get<bool>());else if(v.is_number_integer())sqlite3_bind_int64(raw,i+1,v.get<int64_t>());else if(v.is_number())sqlite3_bind_double(raw,i+1,v.get<double>());else {auto s=v.get<std::string>();sqlite3_bind_text(raw,i+1,s.c_str(),s.size(),SQLITE_TRANSIENT);}}
        std::vector<Json> out;int status;while((status=sqlite3_step(raw))==SQLITE_ROW){Json row=Json::object();for(int i=0;i<sqlite3_column_count(raw);i++){const char* n=sqlite3_column_name(raw,i);auto type=sqlite3_column_type(raw,i);if(type==SQLITE_NULL)row[n]=nullptr;else if(type==SQLITE_INTEGER)row[n]=sqlite3_column_int64(raw,i);else if(type==SQLITE_FLOAT)row[n]=sqlite3_column_double(raw,i);else row[n]=reinterpret_cast<const char*>(sqlite3_column_text(raw,i));}row.erase("deleted");if(row.contains("locked"))row["locked"]=row["locked"].get<int>()!=0;out.push_back(std::move(row));}if(status!=SQLITE_DONE)throw std::runtime_error(sqlite3_errmsg(db_));return out;
    }
public:
    explicit Store(const std::string& root):root_(absolute(root)){fs::create_directories(root_+"/bundles");fs::create_directories(root_+"/samples");if(sqlite3_open((root_+"/voices.sqlite3").c_str(),&db_)!=SQLITE_OK)throw std::runtime_error("Cannot open voice database");sqlite3_busy_timeout(db_,5000);execute("PRAGMA journal_mode=WAL; CREATE TABLE IF NOT EXISTS voices (id TEXT PRIMARY KEY,name TEXT,description TEXT,bundle_path TEXT NOT NULL,sample_path TEXT,source_audio_seconds REAL,source TEXT,created_at TEXT,updated_at TEXT,locked INTEGER DEFAULT 0,deleted INTEGER DEFAULT 0); CREATE TABLE IF NOT EXISTS voice_consents (id TEXT PRIMARY KEY,name TEXT,language TEXT,recording_path TEXT,voice_id TEXT,created_at TEXT,updated_at TEXT,deleted INTEGER DEFAULT 0);");}
    ~Store(){sqlite3_close(db_);}
    Json get(const std::string& table,const std::string& id){std::lock_guard<std::mutex> lock(mutex_);auto rows=table=="voices"?query("SELECT * FROM voices WHERE deleted=0 AND (id=? OR name=?) ORDER BY id=? DESC LIMIT 1",{id,id,id}):query("SELECT * FROM voice_consents WHERE deleted=0 AND id=?",{id});if(rows.empty())return nullptr;rows[0]["object"]=table=="voices"?"audio.voice":"audio.voice_consent";return rows[0];}
    Json list(const std::string& table){std::lock_guard<std::mutex> lock(mutex_);auto rows=query("SELECT * FROM "+table+" WHERE deleted=0 ORDER BY created_at DESC,id DESC");for(auto& r:rows)r["object"]=table=="voices"?"audio.voice":"audio.voice_consent";return Json{{"object","list"},{"data",rows}};}
    Json put(const std::string& table,Json row){std::string columns,placeholders;std::vector<Json> args;row.erase("object");for(auto it=row.begin();it!=row.end();++it){if(!columns.empty()){columns+=',';placeholders+=',';}columns+=it.key();placeholders+='?';args.push_back(it.value());}{std::lock_guard<std::mutex> lock(mutex_);query("INSERT OR REPLACE INTO "+table+" ("+columns+",deleted) VALUES ("+placeholders+",0)",args);}return get(table,row.at("id"));}
    void erase(const std::string& table,const std::string& id){std::lock_guard<std::mutex> lock(mutex_);query("UPDATE "+table+" SET deleted=1,updated_at=? WHERE id=?",{now(),id});}
    void cleanup(const std::string& path){if(path.empty())return;std::error_code error;auto p=fs::weakly_canonical(path,error);if(error||!fs::is_regular_file(p))return;bool owned=false;for(const auto& sub:{"bundles","samples"}){auto root=fs::weakly_canonical(root_+"/"+sub);auto rel=p.lexically_relative(root);if(!rel.empty()&&*rel.begin()!="..")owned=true;}if(!owned)return;std::lock_guard<std::mutex> lock(mutex_);if(query("SELECT 1 FROM voices WHERE deleted=0 AND (bundle_path=? OR sample_path=?) UNION ALL SELECT 1 FROM voice_consents WHERE deleted=0 AND recording_path=?",{p.string(),p.string(),p.string()}).empty())fs::remove(p,error);}
    const std::string& root()const{return root_;}
};
class Service {
public:
    HttpConfig config;Store store;std::unique_ptr<NativeRuntime> runtime;
    std::mutex queue_mutex,cache_mutex;std::condition_variable changed;std::deque<uint64_t> waiting;bool running=false;Json current=nullptr,recent=Json::array();uint64_t submitted=0,completed=0,failed=0,rejected=0;Clock::time_point current_start;
    std::list<std::pair<std::string,NativeVoice>> cache;
    explicit Service(const HttpConfig& c):config(c),store(c.store) {}
    NativeRuntime& engine(){if(!runtime)runtime=std::make_unique<NativeRuntime>(config.model,config.frontend);return *runtime;}
    template<class F> auto job(const std::string& kind,const std::string& label,F fn)->decltype(fn()) {
        auto start=Clock::now();uint64_t id;
        {std::unique_lock<std::mutex> lock(queue_mutex);if(waiting.size()>=config.queue_size){rejected++;throw APIError(429,"Waiting queue is full");}id=++submitted;waiting.push_back(id);changed.wait(lock,[&]{return !running&&waiting.front()==id;});waiting.pop_front();running=true;current_start=Clock::now();current=Json{{"id",id},{"kind",kind},{"label",label},{"status","running"}};}
        auto finish=[&](bool ok,const std::string& error,double audio=0){std::lock_guard<std::mutex> lock(queue_mutex);auto record=current;record["status"]=ok?"completed":"failed";double seconds=std::chrono::duration<double>(Clock::now()-start).count();record["elapsed_seconds"]=seconds;record["audio_seconds"]=audio;record["rtf"]=audio>0?seconds/audio:0;if(!error.empty())record["error"]=error;recent.insert(recent.begin(),record);if(recent.size()>32)recent.erase(recent.end()-1);if(ok)completed++;else failed++;running=false;current=nullptr;changed.notify_all();};
        try{auto result=fn();double audio=0;if constexpr(std::is_same_v<decltype(result),Json>)audio=result.value("audio_seconds",0.0);finish(true,"",audio);return result;}catch(const std::exception& e){finish(false,e.what());throw;}
    }
    Json status(){std::lock_guard<std::mutex> lock(queue_mutex);auto active=current;if(running)active["elapsed_seconds"]=std::chrono::duration<double>(Clock::now()-current_start).count();return Json{{"status","ok"},{"backend","native C++ Metal"},{"runtime","C++"},{"model_version","2.5"},{"model_bundle",absolute(config.model)},{"voice_store",store.root()},{"web_enabled",config.web},{"web_auth_required",!config.webkey.empty()},{"configured_tts_concurrency",config.tts_concurrency},{"configured_clone_concurrency",config.clone_concurrency},{"effective_executor_concurrency",1},{"queue",{{"max_waiting",config.queue_size},{"waiting",waiting.size()},{"running",running},{"current",active}}},{"totals",{{"submitted",submitted},{"completed",completed},{"failed",failed},{"rejected",rejected}}},{"recent",recent}};}
    void remember(const std::string& id,const NativeVoice& voice){std::lock_guard<std::mutex> lock(cache_mutex);cache.remove_if([&](const auto& e){return e.first==id;});cache.emplace_front(id,voice);while(cache.size()>config.voice_cache_size)cache.pop_back();}
    NativeVoice resolve(Json id){if(id.is_object())id=id.value("id",Json(nullptr));if(id.is_null()||id==""){auto records=store.list("voices").at("data");if(records.empty())throw APIError(400,"voice not found; clone a reference first");id=records.back().at("id");}if(!id.is_string())throw APIError(400,"voice must be ID, path or object with id");auto record=store.get("voices",id);if(!record.is_null()){const auto path=record.at("bundle_path").get<std::string>();const auto key=path+":"+std::to_string(static_cast<int64_t>(fs::last_write_time(path).time_since_epoch().count()));{std::lock_guard<std::mutex> lock(cache_mutex);for(auto it=cache.begin();it!=cache.end();++it)if(it->first==key){auto v=it->second;cache.splice(cache.begin(),cache,it);return v;}}auto v=read_native_voice(path);remember(key,v);return v;}
        {std::lock_guard<std::mutex> lock(cache_mutex);for(auto it=cache.begin();it!=cache.end();++it)if(it->first==id.get<std::string>()){auto v=it->second;cache.splice(cache.begin(),cache,it);return v;}}
        return read_native_voice(absolute(id));
    }
    Json create(const NativeVoice& voice,const std::string& name,const std::string& description,const std::string& source="clone",const std::string& sample="",std::string id=""){
        if(id.empty())id=identity("voice");auto dest=store.root()+"/bundles/"+id+".pt";auto temp=dest+".tmp-"+identity("file");save_native_voice(temp,voice);fs::rename(temp,dest);auto old=store.get("voices",id);auto stamp=now();return store.put("voices",Json{{"id",id},{"name",name.empty()?id:name},{"description",description},{"bundle_path",dest},{"sample_path",sample},{"source_audio_seconds",voice.audio.size()/22050.0},{"source",source},{"created_at",old.is_null()?Json(stamp):old["created_at"]},{"updated_at",stamp},{"locked",false}});
    }
};
Json body(const httplib::Request& req){if(req.body.size()>1024*1024)throw APIError(400,"JSON request too large");auto j=Json::parse(req.body,nullptr,false);if(!j.is_object())throw APIError(400,"Expected JSON object");return j;}
void json_response(httplib::Response& res,const Json& j){res.set_content(j.dump(),"application/json");}
std::string form(const httplib::Request& req,const std::string& name,const std::string& def=""){return req.form.has_field(name)?req.form.get_field(name):def;}
std::pair<std::string,std::string> upload(const httplib::Request& req,const fs::path& dir,const std::vector<std::string>& names){std::optional<httplib::FormData> file;for(const auto& name:names)if(req.form.has_file(name)){if(file)throw APIError(400,"Provide only one upload");file=req.form.get_file(name);}if(!file||file->content.empty())throw APIError(400,"Missing audio_sample or recording");if(file->content.size()>128*1024*1024)throw APIError(400,"Upload too large");auto ext=fs::path(file->filename).extension().string();if(ext.size()>10||ext.find_first_not_of(".abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")!=std::string::npos)ext=".wav";auto path=(dir/(identity("upload")+ext)).string();write_file(path,file->content);return {path,ext};}
std::atomic<bool> stopped{false};void stop_signal(int){stopped=true;}
}
int run_http(const HttpConfig& config){
    if(config.queue_size<1||config.queue_size>1024||config.voice_cache_size<1||config.voice_cache_size>1000||config.tts_concurrency<1||config.tts_concurrency>64||config.clone_concurrency<1||config.clone_concurrency>64)throw std::invalid_argument("Invalid queue/cache/concurrency size");Service service(config);httplib::Server server;
    server.new_task_queue=[&]{return new httplib::ThreadPool(std::min(64u,std::max(32u,config.queue_size+8)));};server.set_payload_max_length(128*1024*1024);server.set_read_timeout(600,0);server.set_write_timeout(600,0);
    auto handler=[&](const httplib::Request& req,httplib::Response& res){
        try {
            std::string path=req.path;const auto& method=req.method;Json parsed;bool is_json=req.get_header_value("Content-Type").find("application/json")!=std::string::npos;
            auto get_body=[&]()->Json{if(parsed.is_null())parsed=body(req);return parsed;};
            if(path=="/"&&method=="GET"){if(!config.web)throw APIError(404,"Web UI disabled");res.set_redirect("/web",307);return;}
            if(path=="/web"||path=="/web/"){if(!config.web)throw APIError(404,"Web UI disabled");if(method!="GET")throw APIError(405,"Method not allowed");res.set_content(read_file(config.web_file),"text/html");return;}
            if(path.rfind("/web/api/",0)==0){if(!config.web)throw APIError(404,"Web UI disabled");auto key=req.get_header_value("X-MTTS-Web-Key");auto auth=req.get_header_value("Authorization");if(auth.rfind("Bearer ",0)==0)key=auth.substr(7);if(is_json){auto j=get_body();if(key.empty())key=j.value("key",std::string());}if(!config.webkey.empty()&&key!=config.webkey)throw APIError(401,"Invalid web key");if(path=="/web/api/login"){if(method!="POST")throw APIError(405,"Method not allowed");json_response(res,Json{{"ok",true}});return;}path=path.substr(8);}
            if((path=="/health"||path=="/v1/health")&&method=="GET"){json_response(res,Json{{"status","ok"}});return;}
            if((path=="/api/status"||path=="/status")&&method=="GET"){json_response(res,service.status());return;}
            if((path=="/v1/audio/speech"||path=="/speech")&&method=="POST"){
                auto j=get_body();if(!j.contains("input")||!j["input"].is_string()||j["input"].get<std::string>().find_first_not_of(" \t\r\n")==std::string::npos)throw APIError(400,"input must contain text");if(j.value("response_format",std::string("wav"))!="wav")throw APIError(400,"Only WAV is supported");bool output=j.contains("output");std::string destination;if(output){if(!j["output"].is_string()||j["output"]=="")throw APIError(400,"Invalid output path");destination=absolute(j["output"]);}TempDir temp;if(!output)destination=(temp.path/"speech.wav").string();auto result=service.job("tts",j.at("input"),[&]{auto v=service.resolve(j.value("voice",Json(nullptr)));return service.engine().synthesize(v,j.at("input"),destination,j);});if(output&&!as_bool(j.value("stream",Json(nullptr))))json_response(res,result);else res.set_content(read_file(destination),"audio/wav");return;
            }
            if((path=="/v1/audio/client_voices"||path=="/v1/audio/remote_voices"||path=="/client_voices")&&method=="POST"){
                auto id=form(req,"voice_id",identity("voice"));if(id.empty()||id.size()>128||id.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")!=std::string::npos||id=="."||id=="..")throw APIError(400,"Invalid voice_id");bool persist=as_bool(form(req,"persist","false"));auto old=persist?service.store.get("voices",id):Json(nullptr);if(!old.is_null()&&old.at("locked").get<bool>()){res.set_content(read_file(old.at("bundle_path")),"application/octet-stream");res.set_header("X-Voice-Id",id);return;}
                bool pt=req.form.has_file("pt");if(pt)for(const auto& name:{"audio_sample","recording","file","audio"})if(req.form.has_file(name))throw APIError(400,"Provide only one of audio_sample or pt");TempDir temp;auto file=upload(req,temp.path,pt?std::vector<std::string>{"pt"}:std::vector<std::string>{"audio_sample","recording","file","audio"});auto voice=service.job(pt?"import":"clone",id,[&]{return pt?read_native_voice(file.first):service.engine().clone(file.first);});if(persist)service.create(voice,form(req,"consent"),form(req,"description"),pt?"import":"clone","",id);else service.remember(id,voice);
                if(pt)json_response(res,Json{{"status","ok"},{"voice_id",id},{"persisted",persist},{"cached",!persist}});else {auto path=(temp.path/"voice.pt").string();save_native_voice(path,voice);res.set_content(read_file(path),"application/octet-stream");res.set_header("X-Voice-Id",id);}return;
            }
            std::string prefix;for(const auto& p:{"/api/voices","/v1/audio/voices","/voices"})if(path==p||path.rfind(std::string(p)+"/",0)==0){prefix=p;break;}
            if(!prefix.empty()){
                auto id=path.substr(prefix.size());if(!id.empty())id.erase(0,1);
                if(id.empty()){
                    if(method=="GET"){json_response(res,service.store.list("voices"));return;}
                    if(method!="POST")throw APIError(405,"Method not allowed");NativeVoice voice;std::string name,description,source;
                    if(is_json){auto j=get_body();if(!j.contains("bundle_path")||!j["bundle_path"].is_string())throw APIError(400,"Missing bundle_path");voice=read_native_voice(absolute(j["bundle_path"]));name=j.value("name",std::string());description=j.value("description",std::string());source="import";}
                    else {TempDir temp;auto file=upload(req,temp.path,{"audio_sample","recording","file","audio"});name=form(req,"name");description=form(req,"description");voice=service.job("clone",name,[&]{return service.engine().clone(file.first);});source="clone";}
                    json_response(res,service.create(voice,name,description,source));return;
                }
                std::string suffix;if(id.size()>13&&id.substr(id.size()-13)=="/source-audio"){id.resize(id.size()-13);suffix="audio";}else if(id.size()>5&&id.substr(id.size()-5)=="/lock"){id.resize(id.size()-5);suffix="lock";}auto row=service.store.get("voices",id);if(row.is_null())throw APIError(404,"Voice not found");
                if(suffix=="audio"&&method=="GET"){TempDir temp;auto voice=read_native_voice(row.at("bundle_path"));auto path=(temp.path/"preview.wav").string();save_wave(path,voice.audio);res.set_content(read_file(path),"audio/wav");return;}
                if(suffix=="lock"&&method=="POST"){auto j=get_body();row["locked"]=as_bool(j.value("locked",Json(nullptr)),true);row["updated_at"]=now();json_response(res,service.store.put("voices",row));return;}
                if(!suffix.empty())throw APIError(405,"Method not allowed");if(method=="GET"){json_response(res,row);return;}
                if(method=="PATCH"||method=="PUT"){auto j=get_body();for(const auto& name:{"name","description","bundle_path","sample_path"})if(j.contains(name)){if(!j[name].is_string())throw APIError(400,"Invalid voice field");row[name]=j[name];}if(j.contains("bundle_path")){auto p=absolute(row["bundle_path"]);auto v=read_native_voice(p);row["bundle_path"]=p;row["source_audio_seconds"]=v.audio.size()/22050.0;}row["updated_at"]=now();json_response(res,service.store.put("voices",row));return;}
                if(method=="DELETE"){if(row.at("locked").get<bool>())throw APIError(409,"Voice is locked");service.store.erase("voices",row["id"]);service.store.cleanup(row.value("bundle_path",std::string()));service.store.cleanup(row.value("sample_path",std::string()));json_response(res,Json{{"deleted",true}});return;}throw APIError(405,"Method not allowed");
            }
            for(const auto& p:{"/v1/audio/voice_consents","/voice_consents"})if(path==p||path.rfind(std::string(p)+"/",0)==0){prefix=p;break;}
            if(!prefix.empty()){
                auto id=path.substr(prefix.size());if(!id.empty())id.erase(0,1);
                if(id.empty()){
                    if(method=="GET"){json_response(res,service.store.list("voice_consents"));return;}if(method!="POST")throw APIError(405,"Method not allowed");Json j;std::string recording;bool managed=!is_json;
                    if(is_json){j=get_body();if(!j.contains("recording_path")||!j["recording_path"].is_string())throw APIError(400,"Missing recording_path");recording=absolute(j["recording_path"]);if(!fs::is_regular_file(recording))throw APIError(400,"Recording not found");}
                    else {auto file=upload(req,service.store.root()+"/samples",{"recording","audio_sample","file","audio"});recording=file.first;j=Json{{"name",form(req,"name")},{"language",form(req,"language","zh")},{"create_voice",as_bool(form(req,"create_voice","true"))}};}
                    auto stamp=now();Json row{{"id",identity("consent")},{"name",j.value("name",std::string())},{"language",j.value("language",std::string("zh"))},{"recording_path",recording},{"voice_id",""},{"created_at",stamp},{"updated_at",stamp}};
                    try{if(as_bool(j.value("create_voice",Json(nullptr)),true)){auto voice=service.job("clone",row["name"],[&]{return service.engine().clone(recording);});auto record=service.create(voice,row["name"],"created from voice consent","clone",recording);row["voice_id"]=record["id"];}json_response(res,service.store.put("voice_consents",row));}catch(...){if(managed)service.store.cleanup(recording);throw;}return;
                }
                auto row=service.store.get("voice_consents",id);if(row.is_null())throw APIError(404,"Consent not found");if(method=="GET"){json_response(res,row);return;}if(method=="PATCH"||method=="PUT"){auto j=get_body();for(const auto& name:{"name","language"})if(j.contains(name)){if(!j[name].is_string())throw APIError(400,"Invalid consent field");row[name]=j[name];}row["updated_at"]=now();json_response(res,service.store.put("voice_consents",row));return;}
                if(method=="DELETE"){service.store.erase("voice_consents",row["id"]);service.store.cleanup(row.value("recording_path",std::string()));json_response(res,Json{{"deleted",true}});return;}throw APIError(405,"Method not allowed");
            }
            std::set<std::string> known{"/health","/v1/health","/api/status","/status","/v1/audio/speech","/speech","/v1/audio/client_voices","/v1/audio/remote_voices","/client_voices"};throw APIError(known.count(path)?405:404,"Unknown endpoint or method");
        }catch(const APIError& e){res.status=e.status;json_response(res,Json{{"error",{{"message",e.what()}}}});}catch(const std::invalid_argument& e){res.status=400;json_response(res,Json{{"error",{{"message",e.what()}}}});}catch(const Json::exception& e){res.status=400;json_response(res,Json{{"error",{{"message",e.what()}}}});}catch(const std::exception& e){res.status=500;json_response(res,Json{{"error",{{"message",e.what()}}}});}
        res.set_header("Access-Control-Allow-Origin","*");
    };
    server.set_post_routing_handler([](const auto&,auto& res){res.set_header("Access-Control-Allow-Origin","*");});
    server.Options(".*",[](const auto&,auto& res){res.status=204;res.set_header("Access-Control-Allow-Origin","*");res.set_header("Access-Control-Allow-Methods","GET,POST,PATCH,PUT,DELETE,HEAD,OPTIONS");res.set_header("Access-Control-Allow-Headers","*");});
    server.Get(".*",handler);server.Post(".*",handler);server.Patch(".*",handler);server.Put(".*",handler);server.Delete(".*",handler);
    if(config.seed_example&&service.store.get("voices","voice_demo").is_null()){auto voice=service.job("clone","Official example voice",[&]{return service.engine().clone(config.example_audio);});auto row=service.create(voice,"IndexTTS 2.5 demo","Official example voice","example","","voice_demo");row["locked"]=true;service.store.put("voices",row);}
    stopped=false;std::signal(SIGINT,stop_signal);std::signal(SIGTERM,stop_signal);std::thread watcher([&]{while(!stopped){std::this_thread::sleep_for(std::chrono::milliseconds(100));}server.stop();});
    std::cout<<"IndexTTS 2.5 native: http://"<<config.host<<':'<<config.port<<(config.web?"/web":"/health")<<std::endl;bool ok=server.listen(config.host,config.port);stopped=true;watcher.join();if(!ok)throw std::runtime_error("Cannot listen on requested host/port");return 0;
}
}
