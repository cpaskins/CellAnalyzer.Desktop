using CellAnalyzer.Desktop.Models;
using CellAnalyzer.Desktop.Services;
using Microsoft.Win32;
using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;

namespace CellAnalyzer.Desktop
{
    public partial class PipelineWindow : Window
    {
        private readonly MainWindow _mainWindow;
        private readonly ObservableCollection<PipelineItem> _items = new();
        private bool _isRunning;

        public PipelineWindow(MainWindow mainWindow)
        {
            _mainWindow = mainWindow;
            InitializeComponent();
            ResultsGrid.ItemsSource = _items;
            UpdateProgressDisplay(0, 0);
        }

        // ── Image management ──────────────────────────────────────────────

        private void AddImages_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog
            {
                Filter = "Image Files|*.png;*.jpg;*.jpeg;*.bmp;*.tif",
                Multiselect = true,
                Title = "Select Images for Pipeline"
            };

            if (dialog.ShowDialog() != true) return;

            foreach (var path in dialog.FileNames)
                TryAddImage(path);

            UpdateProgressDisplay(0, _items.Count);
        }

        private void AddFolder_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFolderDialog
            {
                Title = "Select a folder containing images"
            };

            if (dialog.ShowDialog() != true) return;

            var extensions = new[] { ".png", ".jpg", ".jpeg", ".bmp", ".tif" };
            foreach (var file in Directory.GetFiles(dialog.FolderName))
            {
                if (Array.Exists(extensions, ext =>
                    string.Equals(ext, Path.GetExtension(file), StringComparison.OrdinalIgnoreCase)))
                    TryAddImage(file);
            }

            UpdateProgressDisplay(0, _items.Count);
        }

        private void TryAddImage(string path)
        {
            // Deduplicate
            foreach (var existing in _items)
                if (string.Equals(existing.ImagePath, path, StringComparison.OrdinalIgnoreCase))
                    return;

            _items.Add(new PipelineItem
            {
                Index = _items.Count + 1,
                ImagePath = path,
                Status = PipelineStatus.Pending
            });
        }

        private void RemoveSelected_Click(object sender, RoutedEventArgs e)
        {
            if (_isRunning) return;
            if (ResultsGrid.SelectedItem is not PipelineItem item) return;

            _items.Remove(item);
            RenumberItems();
            UpdateProgressDisplay(0, _items.Count);
            PreviewOriginal.Source = null;
            PreviewOverlay.Source = null;
        }

        private void ClearAll_Click(object sender, RoutedEventArgs e)
        {
            if (_isRunning) return;
            _items.Clear();
            PreviewOriginal.Source = null;
            PreviewOverlay.Source = null;
            UpdateProgressDisplay(0, 0);
        }

        private void RenumberItems()
        {
            for (int i = 0; i < _items.Count; i++)
                _items[i].Index = i + 1;
        }

        // ── Preview ───────────────────────────────────────────────────────

        private void ResultsGrid_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (ResultsGrid.SelectedItem is not PipelineItem item) return;

            PreviewOriginal.Source = File.Exists(item.ImagePath)
                ? LoadBitmap(item.ImagePath)
                : null;

            PreviewOverlay.Source = item.OverlayPath != null && File.Exists(item.OverlayPath)
                ? LoadBitmap(item.OverlayPath)
                : null;
        }

        // ── Pipeline execution ────────────────────────────────────────────

        private async void RunPipeline_Click(object sender, RoutedEventArgs e)
        {
            if (_isRunning) return;

            if (_items.Count == 0)
            {
                MessageBox.Show("Add images before running the pipeline.", "No Images",
                    MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            AnalysisParameters p;
            try
            {
                p = _mainWindow.GetCurrentParameters();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Parameter error: {ex.Message}", "Invalid Parameters",
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            _isRunning = true;
            RunButton.IsEnabled = false;

            // Create a shared output directory for this pipeline run
            string runsRoot = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "CellAnalyzer", "runs");

            string pipelineDir = Path.Combine(runsRoot,
                "pipeline_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
            Directory.CreateDirectory(pipelineDir);

            // Write shared params.json once
            string paramsJson = Path.Combine(pipelineDir, "params.json");
            File.WriteAllText(paramsJson,
                JsonSerializer.Serialize(p, new JsonSerializerOptions { WriteIndented = true }));

            // Reset all items
            foreach (var item in _items)
            {
                item.Status = PipelineStatus.Pending;
                item.CellCount = null;
                item.TotalArea = null;
                item.MeanArea = null;
                item.OverlayPath = null;
                item.ErrorMessage = null;
            }

            int total = _items.Count;
            int done = 0;
            UpdateProgressDisplay(done, total);

            foreach (var item in _items)
            {
                item.Status = PipelineStatus.Running;
                RefreshPreviewIfSelected(item);

                try
                {
                    string safeName = SanitizeFileName(Path.GetFileNameWithoutExtension(item.ImagePath));
                    string itemDir = Path.Combine(pipelineDir, $"{item.Index:D3}_{safeName}");
                    Directory.CreateDirectory(itemDir);

                    string outputJson = Path.Combine(itemDir, "result.json");

                    await Task.Run(() => PythonRunner.Run(item.ImagePath, outputJson, paramsJson));

                    string jsonText = File.ReadAllText(outputJson);
                    using var doc = JsonDocument.Parse(jsonText);

                    item.CellCount = doc.RootElement
                        .GetProperty("counts").GetProperty("cell_count").GetInt32();
                    item.TotalArea = doc.RootElement
                        .GetProperty("areas").GetProperty("total_contour_area").GetInt32();
                    item.MeanArea = doc.RootElement
                        .GetProperty("areas").GetProperty("mean_contour_area").GetDouble();

                    string overlayFile = doc.RootElement
                        .GetProperty("images").GetProperty("overlay").GetString()!;
                    item.OverlayPath = Path.Combine(itemDir, overlayFile);
                    item.RunDir = itemDir;
                    item.Status = PipelineStatus.Done;
                }
                catch (Exception ex)
                {
                    item.Status = PipelineStatus.Error;
                    item.ErrorMessage = ex.Message;
                }

                done++;
                UpdateProgressDisplay(done, total);
                RefreshPreviewIfSelected(item);
            }

            _isRunning = false;
            RunButton.IsEnabled = true;

            int errors = 0;
            foreach (var item in _items)
                if (item.Status == PipelineStatus.Error) errors++;

            string summary = errors == 0
                ? $"All {done} image(s) processed successfully."
                : $"{done - errors} of {done} image(s) succeeded; {errors} failed.";

            MessageBox.Show($"{summary}\n\nOutput saved to:\n{pipelineDir}",
                "Pipeline Complete", MessageBoxButton.OK, MessageBoxImage.Information);
        }

        // ── Helpers ───────────────────────────────────────────────────────

        private void RefreshPreviewIfSelected(PipelineItem item)
        {
            if (ResultsGrid.SelectedItem == item)
                ResultsGrid_SelectionChanged(this, null!);
        }

        private void UpdateProgressDisplay(int done, int total)
        {
            ProgressText.Text = $"{done} / {total}";
            PipelineProgressBar.Maximum = total == 0 ? 1 : total;
            PipelineProgressBar.Value = done;
        }

        private static string SanitizeFileName(string name)
        {
            foreach (char c in Path.GetInvalidFileNameChars())
                name = name.Replace(c, '_');
            return name.Length > 60 ? name[..60] : name;
        }

        private static BitmapImage LoadBitmap(string path)
        {
            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.CacheOption = BitmapCacheOption.OnLoad;
            bmp.UriSource = new Uri(path);
            bmp.EndInit();
            bmp.Freeze();
            return bmp;
        }
    }
}
