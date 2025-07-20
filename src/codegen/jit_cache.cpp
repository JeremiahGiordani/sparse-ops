#include "jit_cache.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <mutex>

namespace fs = std::filesystem;

/* simple thread‑safe singleton for the cache directory */
static fs::path cache_dir()
{
    static fs::path dir = []{
        fs::path p = fs::path(std::getenv("HOME")) / ".cache" / "sparseops";
        fs::create_directories(p);
        return p;
    }();
    return dir;
}

static std::mutex cache_mu;

/*---------------------------------------------------------------*/
KernelFn get_or_build_kernel(const std::string& key,
                             const std::string& cpp,
                             const std::string& symbol)
{
    std::lock_guard<std::mutex> lk(cache_mu);

    fs::path so_path = cache_dir() / (key + ".so");
    // enable manually disabling cache hits
    if (fs::exists(so_path) && !std::getenv("SPARSEOPS_NO_CACHE")) {
        if (std::getenv("SPARSEOPS_VERBOSE"))
            std::cerr << "[sparseops] HIT  \"" << so_path << "\"\n";
        return reinterpret_cast<KernelFn>(
            dlsym(dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL |
                                           RTLD_DEEPBIND |
                                           RTLD_NODELETE),
                  symbol.c_str()));
    }

    /* ---------- write temp cpp ---------- */
    fs::path  cpp_path = cache_dir() / (key + ".cpp");
    std::ofstream(cpp_path) << cpp;

    fs::path  so_build = cache_dir() / (key + ".build.so");

    /* ---------- build cmd ---------- */
    const char* cxx       = std::getenv("CXX") ? std::getenv("CXX") : "g++";
    const char* cxxflags  = std::getenv("CXXFLAGS") ? std::getenv("CXXFLAGS") : "-O3 -march=native";
    std::ostringstream cmd;
    cmd << cxx << " -std=c++17 -shared -fPIC " << cxxflags << ' ';

    if (std::getenv("PROFILE_MASKS"))          // pass macro to JIT compile
        cmd << "-DPROFILE_MASKS ";
    cmd << "-I" << (fs::current_path() / "include") << ' '
        << cpp_path << " -o " << so_build;

    if (std::getenv("SPARSEOPS_VERBOSE"))
        std::cerr << "[sparseops] COMPILE -> \"" << so_path << "\"\n";

    /* ---------- compile ---------- */
    if (std::system(cmd.str().c_str()) != 0)
        throw std::runtime_error("jit_cache: clang compilation failed");

    fs::rename(so_build, so_path);

    /* ---------- load ---------- */
    void* handle = dlopen(so_path.c_str(),
                          RTLD_NOW | RTLD_LOCAL |
                          RTLD_DEEPBIND | RTLD_NODELETE);
    if (!handle)
        throw std::runtime_error(dlerror());

    return reinterpret_cast<KernelFn>(dlsym(handle, symbol.c_str()));
}
