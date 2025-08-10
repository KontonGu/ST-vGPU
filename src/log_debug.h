/*
 * Modified on Sat May 18 2024
 *
 * Author: KontonGu
 * Copyright (c) 2024 TUM - CAPS Cloud
 * Licensed under the Apache License, Version 2.0 (the "License")
 */

#ifndef LOGDEBUG_H
#define LOGDEBUG_H


#define LOG_FILE_PATH "debug.log"


#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <sstream>

enum LogLevel {
    DEBUG,
    INFO,
    ERROR,
    WARNING,
};

class Logger {
public:
    Logger() = default;

    template<typename... Args>
    void log(LogLevel level, const std::string& file, int line, const std::string& formatStr, Args... args) {
        std::string message = formatString(formatStr, args...);
        std::string logMessage = formatLogMessage(level, message, file, line);
        std::cout << logMessage << std::endl;
    }

    template<typename... Args>
    void logf(LogLevel level, const std::string& file, int line, const std::string& formatStr, Args... args) {
        std::string message = formatString(formatStr, args...);
        std::string logMessage = formatLogMessage(level, message, file, line);
        std::ofstream logFile(LOG_FILE_PATH, std::ios::app);
        logFile << logMessage << std::endl;
    }

private:
    std::ofstream logFile;
    std::string getCurrentTime() {
        std::time_t now = std::time(nullptr);
        std::tm* localTime = std::localtime(&now);
        std::ostringstream timeStream;
        timeStream << std::put_time(localTime, "%Y-%m-%d %H:%M:%S");
        return timeStream.str();
    }

    std::string logLevelToString(LogLevel level) {
        switch (level) {
            case DEBUG:   return "DEBUG";
            case INFO:    return "INFO";
            case ERROR:  return "ERROR";
            case WARNING: return "WARNING";
            default:      return "UNKNOWN";
        }
    }

    template<typename... Args>
    std::string formatString(const std::string& formatStr, Args... args) {
        std::ostringstream oss;
        formatRecursive(oss, formatStr, args...);
        return oss.str();
    }

    void formatRecursive(std::ostringstream& oss, const std::string& formatStr) {
        oss << formatStr;
    }

    template<typename T, typename... Args>
    void formatRecursive(std::ostringstream& oss, const std::string& formatStr, T value, Args... args) {
        size_t pos = formatStr.find("{}");
        if (pos != std::string::npos) {
            oss << formatStr.substr(0, pos) << value;
            formatRecursive(oss, formatStr.substr(pos + 2), args...);
        } else {
            oss << formatStr; // In case there are more arguments than placeholders
        }
    }

    std::string formatLogMessage(LogLevel level, const std::string& message, const std::string& file, int line) {
        std::ostringstream logStream;
        logStream << "[" << getCurrentTime() << "]"
                  << "[" << logLevelToString(level) << "]"
                  << "[" << file << ":" << line << "] "
                  << message;
        return logStream.str();
    }
};


Logger& getLogger() {
    static Logger logger;
    return logger;
}

    #ifdef ENABLE_DEBUG
        #define LOG_DEBUG(...) getLogger().log(DEBUG, __FILE__, __LINE__,  __VA_ARGS__)
    #else
        #define LOG_DEBUG(...) 
    #endif

#define LOG_INFO(...) getLogger().log(INFO, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) getLogger().log(ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARNING(...) getLogger().log(WARNING, __FILE__, __LINE__, __VA_ARGS__)

#define LOG_DEBUG_FILE(...) getLogger().logf(DEBUG, __FILE__, __LINE__,  __VA_ARGS__)
#define LOG_INFO_FILE(...) getLogger().logf(INFO, __FILE__, __LINE__,  __VA_ARGS__)
#define LOG_ERROR_FILE(...) getLogger().logf(ERROR, __FILE__, __LINE__,  __VA_ARGS__)
#define LOG_WARNING_FILE(...) getLogger().logf(WARNING, __FILE__, __LINE__,  __VA_ARGS__)

#endif