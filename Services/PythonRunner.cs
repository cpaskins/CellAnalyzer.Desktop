using System;
using System.Diagnostics;
using System.IO;

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
            // packaged engine EXE shipped with the app
            string packagedEngine = Path.Combine(BaseDir, "Python", "Engine", "CellAnalyzerEngine.exe");
            if (File.Exists(packagedEngine))
            {
                return (packagedEngine, Path.GetDirectoryName(packagedEngine)!, "");
            }

            // 2) Dev fallback: run via venv python.exe + cli.py
            string pythonExe = FindAbove(Path.Combine("Python", "venv", "Scripts", "python.exe"));
            string cliPath = FindAbove(Path.Combine("Python", "cli.py"));
            string pythonDir = Path.GetDirectoryName(cliPath)!;

            // Arguments will be: "<cliPath>" --image ... --output ... --params ...
            return (pythonExe, pythonDir, $"\"{cliPath}\" ");
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

            string stdout = process.StandardOutput.ReadToEnd();
            string stderr = process.StandardError.ReadToEnd();
            process.WaitForExit();

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

            string stdout = process.StandardOutput.ReadToEnd();
            string stderr = process.StandardError.ReadToEnd();
            process.WaitForExit();

            Debug.WriteLine(stdout);
            Debug.WriteLine(stderr);

            if (process.ExitCode != 0)
                throw new Exception(string.IsNullOrWhiteSpace(stderr) ? "Engine failed with no stderr." : stderr);
        }
    }
}
