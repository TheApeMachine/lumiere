package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/TheApeMachine/lumiere/api"
	"github.com/TheApeMachine/lumiere/config"
)

func main() {
	// Load .env if present so `go run` picks up variables
	_ = loadDotEnv(".env")

	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Ensure required directories exist
	if err := ensureDirectories(cfg); err != nil {
		log.Fatalf("Failed to create directories: %v", err)
	}

	autoLaunch := envIsTrue("AUTO_LAUNCH_AI_SERVICES")
	runDemo := envIsTrue("AUTO_RUN_EXAMPLE")

	var procs []*exec.Cmd
	// Ensure cleanup on panic/normal return
	defer func() {
		if len(procs) > 0 {
			if r := recover(); r != nil {
				fmt.Println("\nCrash detected, shutting down AI services...")
				killProcs(procs)
				panic(r)
			}
			killProcs(procs)
		}
	}()

	// Always handle shutdown signals to terminate child services cleanly
	// We capture the 'procs' slice by reference so processes started later are also killed.
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		fmt.Println("\nShutting down AI services...")
		killProcs(procs)
		os.Exit(0)
	}()
	if autoLaunch {
		fmt.Println("Auto-launching AI services (image, video, audio, creative-director)...")
		var err error
		if err = launchAIServices(&procs); err != nil {
			log.Fatalf("Failed to launch AI services: %v", err)
		}
		// Wait for health checks
		if err := waitForServicesHealth(); err != nil {
			log.Fatalf("AI services not healthy: %v", err)
		}
		// Ensure the pipeline runs in AI mode after services are up
		_ = os.Setenv("USE_AI_SERVICES", "true")
	}

	// Start API server (optionally in background if demo requested)
	fmt.Printf("Starting Lumiere AI Music Video Generator on port %s\n", cfg.ServerPort)
	if runDemo {
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := api.Start(cfg); err != nil {
				log.Fatalf("Failed to start server: %v", err)
			}
		}()
		// Give the server a moment to start listening
		time.Sleep(800 * time.Millisecond)
		if err := runExampleWorkflow(cfg); err != nil {
			log.Printf("Example workflow failed: %v", err)
		}
		// Keep process alive (server goroutine)
		wg.Wait()
	} else {
		if err := api.Start(cfg); err != nil {
			log.Fatalf("Failed to start server: %v", err)
		}
	}

	// Note: child processes will be orphaned if we exit; in production add signal handling to terminate them gracefully
}

func killProcs(cmds []*exec.Cmd) {
	for _, c := range cmds {
		if c != nil && c.Process != nil {
			// Send SIGTERM to process group
			_ = syscall.Kill(-c.Process.Pid, syscall.SIGTERM)
		}
	}
	// Give them a moment to exit, then force kill the groups
	time.Sleep(1200 * time.Millisecond)
	for _, c := range cmds {
		if c != nil && c.Process != nil {
			_ = syscall.Kill(-c.Process.Pid, syscall.SIGKILL)
		}
	}
}

func ensureDirectories(cfg *config.Config) error {
	dirs := []string{cfg.UploadDir, cfg.OutputDir}
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, config.DefaultDirPerms); err != nil {
			return fmt.Errorf("failed to create directory %s: %w", dir, err)
		}
	}
	return nil
}

func envIsTrue(key string) bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv(key)))
	return v == "true" || v == "1" || v == "yes" || v == "y"
}

func loadDotEnv(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		// Support KEY=VALUE
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])
		// Strip optional quotes
		val = strings.Trim(val, "\"'")
		_ = os.Setenv(key, val)
	}
	return nil
}

func getPythonPath() string {
	// Prefer repo venv if present
	venvPython := filepath.Join("venv", "bin", "python")
	if _, err := os.Stat(venvPython); err == nil {
		return venvPython
	}
	return "python3"
}

func launchAIServices(globalProcs *[]*exec.Cmd) error {
	services := []struct {
		name   string
		script string
		req    string
		env    []string
	}{
		{
			name:   "image-generator",
			script: filepath.Join("services", "image-generator", "server.py"),
			req:    filepath.Join("services", "image-generator", "requirements.txt"),
			env:    []string{"IMAGE_SERVICE_PORT=5001"},
		},
		{
			name:   "video-generator",
			script: filepath.Join("services", "video-generator", "server.py"),
			req:    filepath.Join("services", "video-generator", "requirements.txt"),
			env:    []string{"VIDEO_SERVICE_PORT=5002"},
		},
		{
			name:   "audio-analyzer",
			script: filepath.Join("services", "audio-analyzer", "server.py"),
			req:    filepath.Join("services", "audio-analyzer", "requirements.txt"),
			env:    []string{"AUDIO_SERVICE_PORT=5003"},
		},
		{
			name:   "creative-director",
			script: filepath.Join("services", "creative-director", "server.py"),
			req:    filepath.Join("services", "creative-director", "requirements.txt"),
			env:    []string{"PORT=5004"},
		},
	}
	for _, s := range services {
		// Ensure per-service virtual environment exists and use its interpreter
		serviceDir := filepath.Dir(s.script)
		pyFull, err := ensureServiceVenv(serviceDir)
		if err != nil {
			return fmt.Errorf("failed to prepare venv for %s: %w", s.name, err)
		}
		// Install Python dependencies for the service using the same interpreter
		if _, err := os.Stat(s.req); err == nil {
			fmt.Printf("Installing requirements for %s...\n", s.name)
			
			// Special handling for llama-cpp-python on Apple Silicon (needs Metal support)
			// Required for running larger models like Mixtral 8x7B or Llama 3.1 70B
			if s.name == "creative-director" {
				// Check if we're on Apple Silicon
				if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
					fmt.Printf("Detected Apple Silicon - installing llama-cpp-python with Metal support...\n")
					fmt.Printf("This enables efficient inference for large models (Mixtral 8x7B, Llama 3.1 70B, etc.)\n")
					llamaCmd := exec.Command(pyFull, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "llama-cpp-python>=0.2.0")
					llamaCmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
					llamaCmd.Stdout = os.Stdout
					llamaCmd.Stderr = os.Stderr
					llamaEnv := os.Environ()
					llamaEnv = append(llamaEnv, "CMAKE_ARGS=-DLLAMA_METAL=on")
					llamaCmd.Env = llamaEnv
					if err := llamaCmd.Run(); err != nil {
						fmt.Printf("Warning: Failed to install llama-cpp-python with Metal support: %v\n", err)
						fmt.Printf("Continuing with regular installation...\n")
					} else {
						fmt.Printf("Successfully installed llama-cpp-python with Metal support\n")
					}
				}
			}
			
			installCmd := exec.Command(pyFull, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "-r", s.req)
			// Place installer in its own process group so we can terminate it cleanly on signals
			installCmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
			installCmd.Stdout = os.Stdout
			installCmd.Stderr = os.Stderr
			// Set up environment variables for building packages that need openblas (e.g., scipy)
			env := os.Environ()
			// Add PKG_CONFIG_PATH for pkg-config to find openblas (keg-only Homebrew package)
			pkgConfigPath := "/opt/homebrew/opt/openblas/lib/pkgconfig"
			if _, err := os.Stat(pkgConfigPath); err == nil {
				existingPkgConfig := os.Getenv("PKG_CONFIG_PATH")
				if existingPkgConfig != "" {
					env = append(env, fmt.Sprintf("PKG_CONFIG_PATH=%s:%s", pkgConfigPath, existingPkgConfig))
				} else {
					env = append(env, fmt.Sprintf("PKG_CONFIG_PATH=%s", pkgConfigPath))
				}
			}
			// Add LDFLAGS and CPPFLAGS for compilers to find openblas
			env = append(env, "LDFLAGS=-L/opt/homebrew/opt/openblas/lib")
			env = append(env, "CPPFLAGS=-I/opt/homebrew/opt/openblas/include")
			installCmd.Env = append(env, s.env...)
			if err := installCmd.Start(); err != nil {
				return fmt.Errorf("failed to start installer for %s: %w", s.name, err)
			}
			*globalProcs = append(*globalProcs, installCmd)
			if err := installCmd.Wait(); err != nil {
				return fmt.Errorf("failed to install requirements for %s: %w", s.name, err)
			}
		} else if !os.IsNotExist(err) {
			return fmt.Errorf("error checking requirements for %s: %w", s.name, err)
		}

		cmd := exec.Command(pyFull, s.script)
		// Place each child in its own process group so we can signal the group
		cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		cmd.Env = append(os.Environ(), s.env...)
		if err := cmd.Start(); err != nil {
			return fmt.Errorf("failed to start %s: %w", s.name, err)
		}
		fmt.Printf("Started %s (pid=%d)\n", s.name, cmd.Process.Pid)
		*globalProcs = append(*globalProcs, cmd)
		// Stagger slightly to avoid contention on first-time model downloads
		time.Sleep(150 * time.Millisecond)
	}
	return nil
}

// parsePythonVersion extracts the major.minor version from Python version string.
// Example: "Python 3.14.0" -> (3, 14), "Python 3.9.6" -> (3, 9)
func parsePythonVersion(versionStr string) (major, minor int, ok bool) {
	versionStr = strings.TrimSpace(versionStr)
	// Look for pattern like "3.14" or "3.9" in the version string
	parts := strings.Fields(versionStr)
	for _, part := range parts {
		if strings.HasPrefix(part, "3.") {
			versionParts := strings.Split(part, ".")
			if len(versionParts) >= 2 {
				if mj, err := strconv.Atoi(versionParts[0]); err == nil {
					if mn, err := strconv.Atoi(versionParts[1]); err == nil {
						return mj, mn, true
					}
				}
			}
		}
	}
	return 0, 0, false
}

// findPython310Plus finds a Python 3.10+ interpreter, preferring versions that are
// well-supported by scientific packages (3.13, 3.12, 3.11, 3.10).
// Avoids Python 3.14+ as many packages (e.g., numba) don't support it yet.
// Returns the path to the Python interpreter or "python3" as fallback.
func findPython310Plus() string {
	// Try Python versions from 3.13 down to 3.10 (avoid 3.14+ for compatibility)
	// Many packages like numba only support up to 3.13
	for v := 13; v >= 10; v-- {
		py := fmt.Sprintf("python3.%d", v)
		if path, err := exec.LookPath(py); err == nil {
			// Verify it's actually 3.10-3.13
			cmd := exec.Command(path, "--version")
			if output, err := cmd.Output(); err == nil {
				if major, minor, ok := parsePythonVersion(string(output)); ok {
					if major == 3 && minor >= 10 && minor <= 13 {
						return path
					}
				}
			}
		}
	}
	// Fallback to python3, but check version (prefer 3.10-3.13)
	if path, err := exec.LookPath("python3"); err == nil {
		cmd := exec.Command(path, "--version")
		if output, err := cmd.Output(); err == nil {
			if major, minor, ok := parsePythonVersion(string(output)); ok {
				// Prefer 3.10-3.13, but accept 3.14+ as last resort
				if major == 3 && minor >= 10 {
					return path
				}
			}
		}
	}
	return "python3" // Fallback, but may fail later
}

// checkPythonVersion checks if the Python interpreter at pyPath is version 3.10 or higher.
// Prefers versions 3.10-3.13 for better package compatibility.
func checkPythonVersion(pyPath string) (bool, error) {
	cmd := exec.Command(pyPath, "--version")
	output, err := cmd.Output()
	if err != nil {
		return false, err
	}
	major, minor, ok := parsePythonVersion(string(output))
	if !ok {
		return false, nil
	}
	// Accept 3.10+, but prefer 3.10-3.13 (3.14+ may have compatibility issues)
	return major == 3 && minor >= 10, nil
}

// ensureServiceVenv creates a Python virtual environment inside the given service directory if missing
// and returns the absolute path to the venv's python interpreter.
// It ensures the venv uses Python 3.10+ (required for some dependencies like accelerate>=1.11.0).
func ensureServiceVenv(serviceDir string) (string, error) {
	venvDir := filepath.Join(serviceDir, "venv")
	pyPath := filepath.Join(venvDir, "bin", "python")
	
	// Check if venv exists and has Python 3.10-3.13 (avoid 3.14+ for compatibility)
	if _, err := os.Stat(pyPath); err == nil {
		cmd := exec.Command(pyPath, "--version")
		if output, err := cmd.Output(); err == nil {
			if major, minor, ok := parsePythonVersion(string(output)); ok {
				// Accept 3.10-3.13, reject 3.14+ and <3.10
				if major == 3 && minor >= 10 && minor <= 13 {
					return pyPath, nil
				}
				// Python version is incompatible (too old or too new), remove venv
				if major == 3 && minor >= 14 {
					fmt.Printf("Removing venv with Python 3.%d (incompatible with numba/librosa) in %s\n", minor, serviceDir)
				} else {
					fmt.Printf("Removing old venv with Python < 3.10 in %s\n", serviceDir)
				}
				os.RemoveAll(venvDir)
			}
		}
	}
	
	// Find a suitable Python 3.10+ interpreter
	pythonCmd := findPython310Plus()
	
	// Create a dedicated venv for this service using Python 3.10+
	cmd := exec.Command(pythonCmd, "-m", "venv", venvDir)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("create venv failed in %s (using %s): %w", serviceDir, pythonCmd, err)
	}
	
	// Verify the created venv has Python 3.10+
	if ok, err := checkPythonVersion(pyPath); err != nil || !ok {
		return "", fmt.Errorf("created venv in %s does not have Python 3.10+", serviceDir)
	}
	
	// Upgrade pip to reduce resolver issues
	pipUpgrade := exec.Command(pyPath, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
	pipUpgrade.Stdout = os.Stdout
	pipUpgrade.Stderr = os.Stderr
	if err := pipUpgrade.Run(); err != nil {
		return "", fmt.Errorf("upgrade pip failed in %s: %w", serviceDir, err)
	}
	return pyPath, nil
}

func waitForServicesHealth() error {
	urls := []string{
		"http://localhost:5001/health",
		"http://localhost:5002/health",
		"http://localhost:5003/health",
		"http://localhost:5004/healthz",
	}
	// Allow long startup for first-time model downloads (e.g., LTX-Video)
	// Configurable via AI_BOOT_TIMEOUT_SECONDS (default: 900s = 15m)
	bootTimeoutSecs := 900
	if v := strings.TrimSpace(os.Getenv("AI_BOOT_TIMEOUT_SECONDS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			bootTimeoutSecs = n
		}
	}
	deadline := time.Now().Add(time.Duration(bootTimeoutSecs) * time.Second)
	client := &http.Client{Timeout: 2 * time.Second}
	for _, u := range urls {
		fmt.Printf("Waiting for service: %s (timeout %ds)\n", u, int(time.Until(deadline).Seconds()))
		for {
			if time.Now().After(deadline) {
				return fmt.Errorf("timeout waiting for %s", u)
			}
			resp, err := client.Get(u)
			if err == nil && resp.StatusCode == 200 {
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
				fmt.Printf("Service healthy: %s\n", u)
				break
			}
			if resp != nil {
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
			}
			time.Sleep(800 * time.Millisecond)
		}
	}
	return nil
}

func runExampleWorkflow(cfg *config.Config) error {
	prompt := os.Getenv("DEMO_PROMPT")
	if prompt == "" {
		prompt = "Epic cinematic journey through mystical forests with ethereal light"
	}
	audioPath := os.Getenv("DEMO_AUDIO_PATH")
	if audioPath == "" {
		audioPath = "./test_audio.mp3"
	}
	// Validate audio exists
	if _, err := os.Stat(audioPath); err != nil {
		return fmt.Errorf("demo audio not found at %s", audioPath)
	}

	baseURL := fmt.Sprintf("http://localhost:%s", cfg.ServerPort)
	client := &http.Client{Timeout: 30 * time.Second}

	// Create project
	projID, err := httpCreateProject(client, baseURL, prompt, audioPath)
	if err != nil {
		return fmt.Errorf("create project failed: %w", err)
	}
	fmt.Printf("Created project: %s\n", projID)

	// Start processing
	req, _ := http.NewRequest("POST", fmt.Sprintf("%s/api/v1/projects/%s/process", baseURL, projID), nil)
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("start processing failed: %w", err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("start processing returned status %d", resp.StatusCode)
	}
	fmt.Println("Processing started; polling status...")

	// Poll for status changes
	lastStatus := ""
	deadline := time.Now().Add(30 * time.Minute)
	for time.Now().Before(deadline) {
		status, err := httpGetProjectStatus(client, baseURL, projID)
		if err != nil {
			fmt.Printf("Status check error: %v\n", err)
		} else if status != lastStatus {
			fmt.Printf("Project %s status: %s\n", projID, status)
			lastStatus = status
		}

		if status == "completed" || status == "failed" {
			fmt.Printf("Final status: %s\n", status)
			break
		}
		time.Sleep(2 * time.Second)
	}
	return nil
}

func httpCreateProject(client *http.Client, baseURL, prompt, audioPath string) (string, error) {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	// Prompt field
	if err := mw.WriteField("prompt", prompt); err != nil {
		return "", err
	}
	// Audio file
	fw, err := mw.CreateFormFile("audio", filepath.Base(audioPath))
	if err != nil {
		return "", err
	}
	f, err := os.Open(audioPath)
	if err != nil {
		return "", err
	}
	defer f.Close()
	if _, err := io.Copy(fw, f); err != nil {
		return "", err
	}
	mw.Close()

	req, err := http.NewRequest("POST", baseURL+"/api/v1/projects", &buf)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		b, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("create project status %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	// Decode JSON properly
	var pr struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&pr); err != nil {
		return "", fmt.Errorf("decode project response failed: %w", err)
	}
	if pr.ID == "" {
		return "", fmt.Errorf("project id missing in response")
	}
	return pr.ID, nil
}

func httpGetProjectStatus(client *http.Client, baseURL, projectID string) (string, error) {
	req, _ := http.NewRequestWithContext(context.Background(), "GET", fmt.Sprintf("%s/api/v1/projects/%s", baseURL, projectID), nil)
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("status %d", resp.StatusCode)
	}
	var sr struct {
		Status string `json:"status"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		return "", fmt.Errorf("decode status failed: %w", err)
	}
	return sr.Status, nil
}

// (removed naive JSON extractor; using proper json.Decoder above)
