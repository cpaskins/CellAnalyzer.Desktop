using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace CellAnalyzer.Desktop.Services
{
    public static class PythonRunner
    {
        private static string BaseDir => AppDomain.CurrentDomain.BaseDirectory;

        private static string FindAbove(string relativePath)
        {
            var dir = new DirectoryInfo(BaseDir);

            while (dir != null)
            {
                string candidate = Path.Combine(dir.FullName, relativePath);
                if (File.Exists(candidate))
                    return candidate;

                dir = dir.Parent;
            }

            throw new FileNotFoundException($"Could not find {relativePath} above the app directory.");
        }

        private static (string exePath, string workingDir, string argumentsPrefix) ResolveEngine()
        {
            // Prefer developer venv (so installed packages like plotly are available).
            try
            {
                string pythonExe = FindAbove(Path.Combine("Python", "venv", "Scripts", "python.exe"));
                string cliPath = FindAbove(Path.Combine("Python", "cli.py"));
                string pythonDir = Path.GetDirectoryName(cliPath)!;

                // Arguments will be: "<cliPath>" --image ... --output ... --params ...
                return (pythonExe, pythonDir, $"\"{cliPath}\" ");
            }
            catch (FileNotFoundException)
            {
                // venv or cli.py not found; fall back to packaged engine if present
            }

            // packaged engine EXE shipped with the app (PyInstaller COLLECT output)
            string packagedEngine = Path.Combine(BaseDir, "Python", "Engine", "CellAnalyzerEngine", "CellAnalyzerEngine.exe");
            if (File.Exists(packagedEngine))
            {
                return (packagedEngine, Path.GetDirectoryName(packagedEngine)!, "");
            }

            throw new FileNotFoundException("Could not find a Python engine: no venv python.exe/cli.py or packaged engine were found above the app directory.");
        }
        public static string GetDefaultsJson()
        {
            var (exePath, workingDir, prefix) = ResolveEngine();

            var psi = new ProcessStartInfo
            {
                FileName = exePath,
                WorkingDirectory = workingDir,
                Arguments = $"{prefix}--defaults",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = Process.Start(psi);
            if (process == null) throw new Exception("Failed to start engine.");

            // Read stdout and stderr concurrently to prevent pipe-buffer deadlocks
            // (cv2/numpy often emit warnings to stderr during import).
            var stdoutTask = process.StandardOutput.ReadToEndAsync();
            var stderrTask = process.StandardError.ReadToEndAsync();
            process.WaitForExit();
            string stdout = stdoutTask.Result;
            string stderr = stderrTask.Result;

            if (process.ExitCode != 0)
                throw new Exception(string.IsNullOrWhiteSpace(stderr) ? "Engine defaults failed." : stderr);

            return stdout;
        }

        public static void Run(string imagePath, string outputJson, string paramsJsonPath)
        {
            var (exePath, workingDir, prefix) = ResolveEngine();

            Directory.CreateDirectory(Path.GetDirectoryName(outputJson)!);

            var psi = new ProcessStartInfo
            {
                FileName = exePath,
                WorkingDirectory = workingDir,
                Arguments = $"{prefix}--image \"{imagePath}\" --output \"{outputJson}\" --params \"{paramsJsonPath}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = Process.Start(psi);
            if (process == null)
                throw new Exception("Failed to start analysis engine process.");

            // Read stdout and stderr concurrently to prevent pipe-buffer deadlocks.
            var stdoutTask = process.StandardOutput.ReadToEndAsync();
            var stderrTask = process.StandardError.ReadToEndAsync();
            process.WaitForExit();
            string stdout = stdoutTask.Result;
            string stderr = stderrTask.Result;

            Debug.WriteLine(stdout);
            Debug.WriteLine(stderr);

            if (process.ExitCode != 0)
                throw new Exception(string.IsNullOrWhiteSpace(stderr) ? "Engine failed with no stderr." : stderr);
        }
    }
}
