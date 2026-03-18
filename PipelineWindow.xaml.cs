using CellAnalyzer.Desktop.Models;
using CellAnalyzer.Desktop.Services;
using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace CellAnalyzer.Desktop
{
    public partial class PipelineWindow : Window
    {
        private readonly MainWindow _mainWindow;
        private readonly ObservableCollection<PipelineItem> _items = new();
        private bool _isRunning;

        // Preview navigation state
        private int _previewIndex = -1;     // index into _items; -1 = nothing shown
        private bool _updatingPreview;      // re-entrancy guard for grid ↔ nav sync

        // Chart colours (BGR order is for Python; these are WPF Colors)
        private static readonly Color ChartBlue  = Color.FromRgb(0x4B, 0x9C, 0xD3);
        private static readonly Color ChartGreen = Color.FromRgb(0x6B, 0xCB, 0x77);
        private static readonly Color ChartText  = Color.FromRgb(0xCC, 0xCC, 0xCC);
        private static readonly Color ChartLabel = Color.FromRgb(0x77, 0x88, 0x99);
        private static readonly Color ChartBase  = Color.FromRgb(0x33, 0x44, 0x55);

        // Logical canvas dimensions used for all bar charts
        private const double CW = 500, CH = 160;
        private const double CPadTop = 22, CPadBottom = 30, CPadSide = 8;

        public PipelineWindow(MainWindow mainWindow)
        {
            _mainWindow = mainWindow;
            InitializeComponent();
            ResultsGrid.ItemsSource = _items;
            UpdateProgressDisplay(0, 0);
            RefreshDashboard();
        }

        // ── Image management ─────────────────────────────────────────────

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
                    string.Equals(ext, System.IO.Path.GetExtension(file), StringComparison.OrdinalIgnoreCase)))
                    TryAddImage(file);
            }

            UpdateProgressDisplay(0, _items.Count);
        }

        private void TryAddImage(string path)
        {
            foreach (var existing in _items)
                if (string.Equals(existing.ImagePath, path, StringComparison.OrdinalIgnoreCase))
                    return;

            _items.Add(new PipelineItem
            {
                Index     = _items.Count + 1,
                ImagePath = path,
                Status    = PipelineStatus.Pending
            });
        }

        private void RemoveSelected_Click(object sender, RoutedEventArgs e)
        {
            if (_isRunning) return;
            if (ResultsGrid.SelectedItem is not PipelineItem item) return;

            _items.Remove(item);
            RenumberItems();
            UpdateProgressDisplay(0, _items.Count);
            ClearPreview();
            RefreshDashboard();
        }

        private void ClearAll_Click(object sender, RoutedEventArgs e)
        {
            if (_isRunning) return;
            _items.Clear();
            UpdateProgressDisplay(0, 0);
            ClearPreview();
            RefreshDashboard();
        }

        private void RenumberItems()
        {
            for (int i = 0; i < _items.Count; i++)
                _items[i].Index = i + 1;
        }

        // ── Preview navigation ────────────────────────────────────────────

        private void ResultsGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_updatingPreview) return;
            int idx = ResultsGrid.SelectedIndex;
            if (idx >= 0) UpdatePreview(idx);
        }

        private void PrevImage_Click(object sender, RoutedEventArgs e)
        {
            if (_items.Count == 0) return;
            int next = _previewIndex <= 0 ? _items.Count - 1 : _previewIndex - 1;
            UpdatePreview(next);
        }

        private void NextImage_Click(object sender, RoutedEventArgs e)
        {
            if (_items.Count == 0) return;
            int next = _previewIndex >= _items.Count - 1 ? 0 : _previewIndex + 1;
            UpdatePreview(next);
        }

        /// <summary>
        /// Single source of truth for the preview pane. Syncs the DataGrid selection,
        /// image thumbnails, navigation counter, and filename badge.
        /// </summary>
        private void UpdatePreview(int index)
        {
            if (index < 0 || index >= _items.Count) { ClearPreview(); return; }

            _previewIndex = index;
            var item = _items[index];

            // Sync grid selection without triggering another UpdatePreview
            _updatingPreview = true;
            try { ResultsGrid.SelectedIndex = index; }
            finally { _updatingPreview = false; }

            PreviewIndexText.Text   = $"{index + 1} / {_items.Count}";
            PreviewFileNameText.Text = item.FileName;

            PreviewOriginal.Source = File.Exists(item.ImagePath)
                ? LoadBitmap(item.ImagePath)
                : null;

            PreviewOverlay.Source = item.OverlayPath != null && File.Exists(item.OverlayPath)
                ? LoadBitmap(item.OverlayPath)
                : null;
        }

        private void ClearPreview()
        {
            _previewIndex        = -1;
            PreviewOriginal.Source  = null;
            PreviewOverlay.Source   = null;
            PreviewIndexText.Text   = "–";
            PreviewFileNameText.Text = "";
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

            // Create shared output directory
            string runsRoot = System.IO.Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "CellAnalyzer", "runs");

            string pipelineDir = System.IO.Path.Combine(runsRoot,
                "pipeline_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
            Directory.CreateDirectory(pipelineDir);

            string paramsJson = System.IO.Path.Combine(pipelineDir, "params.json");
            File.WriteAllText(paramsJson,
                JsonSerializer.Serialize(p, new JsonSerializerOptions { WriteIndented = true }));

            // Reset all items
            foreach (var item in _items)
            {
                item.Status       = PipelineStatus.Pending;
                item.CellCount    = null;
                item.TotalArea    = null;
                item.MeanArea     = null;
                item.OverlayPath  = null;
                item.ErrorMessage = null;
            }

            int total = _items.Count;
            int done  = 0;
            UpdateProgressDisplay(done, total);
            RefreshDashboard();

            foreach (var item in _items)
            {
                item.Status = PipelineStatus.Running;
                RefreshPreviewIfSelected(item);

                try
                {
                    string safeName = SanitizeFileName(System.IO.Path.GetFileNameWithoutExtension(item.ImagePath));
                    string itemDir  = System.IO.Path.Combine(pipelineDir, $"{item.Index:D3}_{safeName}");
                    Directory.CreateDirectory(itemDir);

                    string outputJson = System.IO.Path.Combine(itemDir, "result.json");

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
                    item.OverlayPath = System.IO.Path.Combine(itemDir, overlayFile);
                    item.RunDir      = itemDir;
                    item.Status      = PipelineStatus.Done;
                }
                catch (Exception ex)
                {
                    item.Status       = PipelineStatus.Error;
                    item.ErrorMessage = ex.Message;
                }

                done++;
                UpdateProgressDisplay(done, total);
                RefreshPreviewIfSelected(item);
                RefreshDashboard();     // update charts + stats after every image
            }

            _isRunning          = false;
            RunButton.IsEnabled = true;

            int errors = _items.Count(i => i.Status == PipelineStatus.Error);
            string summary = errors == 0
                ? $"All {done} image(s) processed successfully."
                : $"{done - errors} of {done} image(s) succeeded; {errors} failed.";

            MessageBox.Show($"{summary}\n\nOutput saved to:\n{pipelineDir}",
                "Pipeline Complete", MessageBoxButton.OK, MessageBoxImage.Information);
        }

        // ── Charts ────────────────────────────────────────────────────────

        /// <summary>Redraws both bar charts and refreshes the stats cards.</summary>
        private void RefreshDashboard()
        {
            var done = _items.Where(i => i.Status == PipelineStatus.Done).ToList();

            DrawBarChart(
                CellCountChart,
                done.Select(i => (double)(i.CellCount ?? 0)).ToList(),
                done.Select(i => System.IO.Path.GetFileNameWithoutExtension(i.ImagePath)).ToList(),
                ChartBlue);

            DrawBarChart(
                AreaChart,
                done.Select(i => i.MeanArea ?? 0.0).ToList(),
                done.Select(i => System.IO.Path.GetFileNameWithoutExtension(i.ImagePath)).ToList(),
                ChartGreen);

            RefreshStats(done);
        }

        private void DrawBarChart(Canvas canvas, List<double> values, List<string> labels, Color barColor)
        {
            canvas.Children.Clear();

            double innerW = CW - CPadSide * 2;
            double innerH = CH - CPadTop - CPadBottom;
            double originY = CPadTop;

            if (values.Count == 0)
            {
                var msg = new TextBlock
                {
                    Text       = "No completed images yet",
                    Foreground = new SolidColorBrush(ChartLabel),
                    FontSize   = 11
                };
                Canvas.SetLeft(msg, CW / 2 - 80);
                Canvas.SetTop(msg,  CH / 2 - 8);
                canvas.Children.Add(msg);
                return;
            }

            double maxVal  = values.Max();
            if (maxVal <= 0) maxVal = 1;

            double slotW     = innerW / values.Count;
            double barW      = Math.Max(slotW * 0.65, 2);
            double barMargin = (slotW - barW) / 2;
            bool   showLabels = values.Count <= 20;

            for (int i = 0; i < values.Count; i++)
            {
                double barH = values[i] / maxVal * innerH;
                double x    = CPadSide + i * slotW + barMargin;
                double y    = originY + innerH - barH;

                // Bar
                var rect = new Rectangle
                {
                    Width   = barW,
                    Height  = Math.Max(barH, 1),
                    Fill    = new SolidColorBrush(barColor),
                    RadiusX = 2,
                    RadiusY = 2
                };
                Canvas.SetLeft(rect, x);
                Canvas.SetTop(rect,  y);
                canvas.Children.Add(rect);

                // Value label above bar
                string valStr = values[i] >= 1000
                    ? $"{values[i] / 1000:F1}k"
                    : $"{values[i]:F0}";

                var valText = new TextBlock
                {
                    Text       = valStr,
                    FontSize   = 8,
                    Foreground = new SolidColorBrush(ChartText)
                };
                Canvas.SetLeft(valText, x + barW / 2 - 8);
                Canvas.SetTop(valText,  y - 13);
                canvas.Children.Add(valText);

                // Filename label beneath bar (rotated, only when few images)
                if (showLabels && i < labels.Count)
                {
                    string lbl = labels[i].Length > 8
                        ? labels[i][..8] + "…"
                        : labels[i];

                    var lblText = new TextBlock
                    {
                        Text                  = lbl,
                        FontSize              = 7,
                        Foreground            = new SolidColorBrush(ChartLabel),
                        RenderTransformOrigin = new Point(0, 0),
                        RenderTransform       = new RotateTransform(-35)
                    };
                    Canvas.SetLeft(lblText, x + barW / 2);
                    Canvas.SetTop(lblText,  originY + innerH + 3);
                    canvas.Children.Add(lblText);
                }
            }

            // Baseline
            var baseline = new Line
            {
                X1              = CPadSide - 2,
                Y1              = originY + innerH,
                X2              = CPadSide + innerW,
                Y2              = originY + innerH,
                Stroke          = new SolidColorBrush(ChartBase),
                StrokeThickness = 1
            };
            canvas.Children.Add(baseline);
        }

        // ── Stats ─────────────────────────────────────────────────────────

        private void RefreshStats(List<PipelineItem> done)
        {
            int total  = _items.Count;
            int errors = _items.Count(i => i.Status == PipelineStatus.Error);

            StatsProcessed.Text  = $"{done.Count} / {total}";
            StatsErrors.Text     = errors.ToString();

            var cells = done.Where(i => i.CellCount.HasValue).Select(i => (double)i.CellCount!.Value).ToList();
            var areas = done.Where(i => i.MeanArea.HasValue).Select(i => i.MeanArea!.Value).ToList();

            if (cells.Count == 0)
            {
                StatsTotalCells.Text = "–";
                StatsCellMin.Text = StatsCellMax.Text = StatsCellMean.Text = StatsCellStdDev.Text = "–";
            }
            else
            {
                double cellMean   = cells.Average();
                double cellStdDev = Math.Sqrt(cells.Average(c => Math.Pow(c - cellMean, 2)));
                StatsTotalCells.Text  = cells.Sum().ToString("F0");
                StatsCellMin.Text     = cells.Min().ToString("F0");
                StatsCellMax.Text     = cells.Max().ToString("F0");
                StatsCellMean.Text    = cellMean.ToString("F1");
                StatsCellStdDev.Text  = cellStdDev.ToString("F1");
            }

            if (areas.Count == 0)
            {
                StatsAreaMin.Text = StatsAreaMax.Text = StatsAreaMean.Text = StatsAreaStdDev.Text = "–";
            }
            else
            {
                double areaMean   = areas.Average();
                double areaStdDev = Math.Sqrt(areas.Average(a => Math.Pow(a - areaMean, 2)));
                StatsAreaMin.Text    = areas.Min().ToString("F1");
                StatsAreaMax.Text    = areas.Max().ToString("F1");
                StatsAreaMean.Text   = areaMean.ToString("F1");
                StatsAreaStdDev.Text = areaStdDev.ToString("F1");
            }
        }

        // ── Helpers ───────────────────────────────────────────────────────

        private void RefreshPreviewIfSelected(PipelineItem item)
        {
            int idx = _items.IndexOf(item);
            if (idx == _previewIndex)
                UpdatePreview(idx);
        }

        private void UpdateProgressDisplay(int done, int total)
        {
            ProgressText.Text              = $"{done} / {total}";
            PipelineProgressBar.Maximum    = total == 0 ? 1 : total;
            PipelineProgressBar.Value      = done;
        }

        private static string SanitizeFileName(string name)
        {
            foreach (char c in System.IO.Path.GetInvalidFileNameChars())
                name = name.Replace(c, '_');
            return name.Length > 60 ? name[..60] : name;
        }

        private static BitmapImage LoadBitmap(string path)
        {
            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.CacheOption = BitmapCacheOption.OnLoad;
            bmp.UriSource   = new Uri(path);
            bmp.EndInit();
            bmp.Freeze();
            return bmp;
        }
    }
}
