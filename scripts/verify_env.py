"""
Phase 0: Environment & Hardware Verification Probe for ARIA
"""
import sys
import platform
import os
import json
import subprocess
import shutil

def get_cpu_info():
    logical_cores = os.cpu_count() or 0
    # In Windows, we can also query WMIC / PowerShell for physical cores
    try:
        cmd = "powershell -Command \"(Get-CimInstance Win32_Processor).NumberOfCores\""
        out = subprocess.check_output(cmd, shell=True, text=True).strip()
        physical_cores = int(out.splitlines()[0]) if out else logical_cores
    except Exception:
        physical_cores = logical_cores
    return {
        "processor": platform.processor(),
        "logical_cores": logical_cores,
        "physical_cores": physical_cores
    }

def get_ram_info():
    try:
        cmd = "powershell -Command \"Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory\""
        out = subprocess.check_output(cmd, shell=True, text=True).strip()
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        # Typically Header line, separator line, value line
        values = lines[-1].split()
        total_kb = int(values[0])
        free_kb = int(values[1])
        return {
            "total_ram_mb": round(total_kb / 1024, 2),
            "free_ram_mb": round(free_kb / 1024, 2),
            "total_ram_gb": round(total_kb / (1024 * 1024), 2),
            "free_ram_gb": round(free_kb / (1024 * 1024), 2)
        }
    except Exception as e:
        return {"error": str(e)}

def get_gpu_info():
    try:
        cmd = "nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,memory.used --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd, shell=True, text=True).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_detected": True,
            "gpu_name": parts[0],
            "driver_version": parts[1],
            "vram_total_mb": float(parts[2]),
            "vram_free_mb": float(parts[3]),
            "vram_used_mb": float(parts[4])
        }
    except Exception as e:
        return {
            "gpu_detected": False,
            "error": str(e)
        }

def get_audio_devices():
    try:
        cmd = "powershell -Command \"Get-CimInstance Win32_SoundDevice | Select-Object Name, Status\""
        out = subprocess.check_output(cmd, shell=True, text=True).strip()
        return {"devices_summary": out}
    except Exception as e:
        return {"error": str(e)}

def check_ollama():
    ollama_path = shutil.which("ollama")
    version = None
    service_running = False
    
    if ollama_path:
        try:
            out = subprocess.check_output("ollama --version", shell=True, text=True).strip()
            version = out
        except Exception:
            pass
            
    # Check if port 11434 is responding
    import urllib.request
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/")
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                service_running = True
    except Exception:
        service_running = False
        
    return {
        "installed": ollama_path is not None,
        "path": ollama_path,
        "version": version,
        "service_running_11434": service_running
    }

def main():
    print("Gathering ARIA Phase 0 Environment & Hardware Data...")
    
    report = {
        "phase": 0,
        "timestamp": platform.uname().version,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "architecture": platform.architecture()[0]
        },
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "is_64bit": sys.maxsize > 2**32
        },
        "cpu": get_cpu_info(),
        "ram": get_ram_info(),
        "gpu": get_gpu_info(),
        "audio": get_audio_devices(),
        "ollama": check_ollama(),
        "criteria_checks": {}
    }
    
    # Check Acceptance Criteria
    checks = {}
    checks["python_version_gte_3_10"] = sys.version_info >= (3, 10)
    checks["python_64bit"] = sys.maxsize > 2**32
    checks["cuda_gpu_detected"] = report["gpu"].get("gpu_detected", False)
    checks["vram_gte_4000_mb"] = report["gpu"].get("vram_total_mb", 0) >= 4000.0
    checks["audio_devices_present"] = bool(report["audio"].get("devices_summary"))
    checks["ollama_service_operational"] = report["ollama"].get("service_running_11434", False)
    
    report["criteria_checks"] = checks
    
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "phase0_environment_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nReport written to {report_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
