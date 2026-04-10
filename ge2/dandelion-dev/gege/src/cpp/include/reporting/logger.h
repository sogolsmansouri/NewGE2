#pragma once

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <string>
#include <vector>

using std::shared_ptr;
using std::string;

class GegeLogger {
   private:
    shared_ptr<spdlog::sinks::basic_file_sink_mt> trace_sink_;
    shared_ptr<spdlog::sinks::basic_file_sink_mt> debug_sink_;
    shared_ptr<spdlog::sinks::basic_file_sink_mt> info_sink_;
    shared_ptr<spdlog::sinks::basic_file_sink_mt> warn_sink_;
    shared_ptr<spdlog::sinks::basic_file_sink_mt> error_sink_;
    shared_ptr<spdlog::sinks::stdout_color_sink_mt> console_sink_;

   public:
    shared_ptr<spdlog::logger> main_logger_;

    GegeLogger(string model_dir) {
        spdlog::drop_all();

        console_sink_ = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink_->set_level(spdlog::level::info);
        console_sink_->set_pattern("[%x %T.%e] %v");

        bool disable_file_logs = false;
        if (const char *disable_env = std::getenv("GEGE_DISABLE_FILE_LOGS")) {
            disable_file_logs = std::string(disable_env) == "1";
        }

        std::vector<spdlog::sink_ptr> sinks;
        sinks.push_back(console_sink_);

        if (!disable_file_logs) {
            trace_sink_ = std::make_shared<spdlog::sinks::basic_file_sink_mt>(fmt::format("{}/logs/{}.log", model_dir, "trace"), true);
            trace_sink_->set_level(spdlog::level::trace);
            trace_sink_->set_pattern("[%l] [%x %T.%e] [PID:%P TID:%t] [%s:%!:%#] %v");

            debug_sink_ = std::make_shared<spdlog::sinks::basic_file_sink_mt>(fmt::format("{}/logs/{}.log", model_dir, "debug"), true);
            debug_sink_->set_level(spdlog::level::debug);
            debug_sink_->set_pattern("[%l] [%x %T.%e] [PID:%P TID:%t] [%s:%!:%#] %v");

            info_sink_ = std::make_shared<spdlog::sinks::basic_file_sink_mt>(fmt::format("{}/logs/{}.log", model_dir, "info"), true);
            info_sink_->set_level(spdlog::level::info);
            info_sink_->set_pattern("[%l] [%x %T.%e] [PID:%P TID:%t] [%s:%!:%#] %v");

            warn_sink_ = std::make_shared<spdlog::sinks::basic_file_sink_mt>(fmt::format("{}/logs/{}.log", model_dir, "warn"), true);
            warn_sink_->set_level(spdlog::level::warn);
            warn_sink_->set_pattern("[%l] [%x %T.%e] [PID:%P TID:%t] [%s:%!:%#] %v");

            error_sink_ = std::make_shared<spdlog::sinks::basic_file_sink_mt>(fmt::format("{}/logs/{}.log", model_dir, "error"), true);
            error_sink_->set_level(spdlog::level::err);
            error_sink_->set_pattern("[%l] [%x %T.%e] [PID:%P TID:%t] [%g:%s:%!:%#] %v");

            sinks.insert(sinks.begin(), {error_sink_, warn_sink_, info_sink_, debug_sink_, trace_sink_});
        }

        main_logger_ = std::make_shared<spdlog::logger>("GegeLogger", sinks.begin(), sinks.end());
        main_logger_->set_level(spdlog::level::trace);
        spdlog::register_logger(main_logger_);

        spdlog::flush_every(std::chrono::seconds(1));
    }

    void setConsoleLogLevel(spdlog::level::level_enum level) { console_sink_->set_level(level); }
};
