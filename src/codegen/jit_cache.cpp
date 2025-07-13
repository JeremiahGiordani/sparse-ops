#include "jit_cache.hpp"
#include "bcoo16_encoder.hpp"          // for BCOO16Block definition

#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/stat.h>
#include <iostream>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

// *Very* simple hash  →  16-byte hex string (replace w/ SHA-256 later).
static std::string hash_key(const std::string& s) {
    std::size_t h = std::hash<std::string>{}(s);
    std::ostringstream oss;
    oss << std::hex << h;
    return oss.str();
}

static fs::path cache_directory() {
    const char* home = std::getenv("HOME");
    fs::path dir = home ? fs::path(home) / ".cache" / "sparseops"
                        : fs::temp_directory_path() / "sparseops";
    fs::create_directories(dir);
    return dir;
}

static void write_file(const fs::path& p, const std::string& text) {
    std::ofstream ofs(p, std::ios::out | std::ios::trunc);
    if (!ofs) throw std::runtime_error("jit_cache: cannot open " + p.string());
    ofs << text;
}

static bool verbose() {
    return std::getenv("SPARSEOPS_VERBOSE") != nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public API
// ─────────────────────────────────────────────────────────────────────────────
KernelFn get_or_build_kernel(const std::string& key,
                             const std::string& cpp_src,
                             const std::string& func_name,
                             const std::string& clang_flags)
{
    fs::path cache_dir = cache_directory();
    std::string key_hex = hash_key(key);
    fs::path so_path = cache_dir / (key_hex + ".so");

    // 1. Use cached .so if it exists
    if (fs::exists(so_path)) {
        if (verbose()) std::cerr << "[sparseops] HIT  " << so_path << '\n';
        void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle)
            throw std::runtime_error("jit_cache: dlopen failed for " +
                                     so_path.string() + "\n" + dlerror());
        auto sym = reinterpret_cast<KernelFn>(dlsym(handle, func_name.c_str()));
        if (!sym)
            throw std::runtime_error("jit_cache: dlsym failed for " +
                                     func_name + "\n" + dlerror());
        return sym;
    }

    if (verbose()) std::cerr << "[sparseops] COMPILE -> " << so_path << '\n';

    // 2. Compile new kernel
    fs::path tmp_cpp = cache_dir / (key_hex + ".cpp");
    write_file(tmp_cpp, cpp_src);

    fs::path tmp_so  = cache_dir / (key_hex + ".build.so");

    // Choose a compiler: env SPARSEOPS_CXX > clang++ > g++
    const char* cxx =
        std::getenv("SPARSEOPS_CXX") ? std::getenv("SPARSEOPS_CXX") :
        (std::system("command -v clang++ > /dev/null 2>&1") == 0 ? "clang++" :
        "g++");   // assume g++ is present

    std::ostringstream cmd;
    cmd << cxx << " -std=c++17 -shared -fPIC "
        << clang_flags << ' '
        << "-I" << "/home/jg0037/sparse-ops/include" << " "
        << tmp_cpp << " -o " << tmp_so;

    int ret = std::system(cmd.str().c_str());
    if (ret != 0)
        throw std::runtime_error("jit_cache: clang compilation failed");

    // 3. Atomically move into cache path
    fs::rename(tmp_so, so_path);

    // 4. dlopen
    void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle)
        throw std::runtime_error("jit_cache: dlopen failed for " +
                                 so_path.string() + "\n" + dlerror());
    auto sym = reinterpret_cast<KernelFn>(dlsym(handle, func_name.c_str()));
    if (!sym)
        throw std::runtime_error("jit_cache: dlsym failed for " +
                                 func_name + "\n" + dlerror());
    return sym;
}
