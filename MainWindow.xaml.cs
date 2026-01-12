using CellAnalyzer.Desktop.Services;
using Microsoft.Win32;
using System;
using System.Globalization;
using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Media.Imaging;
using CellAnalyzer.Desktop.Models;

namespace CellAnalyzer.Desktop
{
    public partial class MainWindow : Window
    {
        private string? _imagePath;

        public MainWindow()
        {
            InitializeComponent();
            LoadDefaultsFromEngine();
        }

        private void LoadDefaultsFromEngine()
        {
            try
            {
                StatusText.Text = "Status: Loading defaults...";

                string json = PythonRunner.GetDefaultsJson();

                var p = JsonSerializer.Deserialize<AnalysisParameters>(json, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

                if (p == null)
                    throw new Exception("Engine returned invalid defaults.");

                ApplyParametersToUi(p);

                StatusText.Text = "Status: Defaults loaded";
            }
            catch (Exception ex)
            {
                // Fallback strategy if engine fails
                StatusText.Text = "Status: Defaults failed (using UI values)";
                System.Diagnostics.Debug.WriteLine(ex);
            }
        }

        private void LoadImage_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog
            {
                Filter = "Image Files|*.png;*.jpg;*.jpeg;*.bmp;*.tif"
            };

            if (dialog.ShowDialog() == true)
            {
                _imagePath = dialog.FileName;

                OriginalImage.Source = LoadBitmap(_imagePath);
                OverlayImage.Source = null;

                CellCountText.Text = "Cells: -";
                TotalAreaText.Text = "Total Contour Area: -";
                MeanAreaText.Text = "Mean Contour Area: -";
                StatusText.Text = "Status: Image loaded";
            }
        }

        private void Analyze_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(_imagePath))
            {
                MessageBox.Show("Load an image first");
                return;
            }

            try
            {
                StatusText.Text = "Status: Preparing run...";

                // Create run folder so outputs never go in the image folder
                string runsRoot = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    "CellAnalyzer",
                    "runs"
                );

                string runId = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string runDir = Path.Combine(runsRoot, runId);
                Directory.CreateDirectory(runDir);

                string outputJson = Path.Combine(runDir, "result.json");
                string paramsJson = Path.Combine(runDir, "params.json");

                // Build params from UI (with validation)
                var p = BuildParamsFromUi();

                // Write params.json for Python
                File.WriteAllText(
                    paramsJson,
                    JsonSerializer.Serialize(p, new JsonSerializerOptions { WriteIndented = true })
                );

                StatusText.Text = "Status: Analyzing...";

                // Run Python with params
                PythonRunner.Run(_imagePath, outputJson, paramsJson);

                // Read results
                string jsonText = File.ReadAllText(outputJson);
                using var doc = JsonDocument.Parse(jsonText);

                int cellCount = doc.RootElement.GetProperty("counts").GetProperty("cell_count").GetInt32();
                int totalContourArea = doc.RootElement.GetProperty("areas").GetProperty("total_contour_area").GetInt32();
                double meanContourArea = doc.RootElement.GetProperty("areas").GetProperty("mean_contour_area").GetDouble();

                CellCountText.Text = $"Cells: {cellCount}";
                TotalAreaText.Text = $"Total Contour Area: {totalContourArea} µm²";
                MeanAreaText.Text = $"Mean Contour Area: {meanContourArea} µm²";

                // Load overlay path from JSON and display it
                string overlayFile = doc.RootElement.GetProperty("images").GetProperty("overlay").GetString()!;
                string overlayPath = Path.Combine(runDir, overlayFile);

                if (!File.Exists(overlayPath))
                    throw new FileNotFoundException("Overlay image not found", overlayPath);

                OverlayImage.Source = LoadBitmap(overlayPath);

                StatusText.Text = $"Status: Done (saved to {runDir})";
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                StatusText.Text = "Status: Error";
            }
        }

        private void SavePreset_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Build current params from UI (your existing method)
                AnalysisParameters p = BuildParamsFromUi();

                var preset = new ParameterPreset
                {
                    schema_version = 1,
                    name = "Preset",
                    saved_at = DateTime.Now,
                    parameters = p
                };

                var dialog = new SaveFileDialog
                {
                    Title = "Save Parameter Preset",
                    Filter = "CellAnalyzer Preset (*.cellpreset)|*.cellpreset|JSON (*.json)|*.json",
                    DefaultExt = ".cellpreset",
                    AddExtension = true,
                    FileName = "preset.cellpreset"
                };

                if (dialog.ShowDialog() != true)
                    return;

                string json = JsonSerializer.Serialize(preset, new JsonSerializerOptions
                {
                    WriteIndented = true
                });

                File.WriteAllText(dialog.FileName, json);

                StatusText.Text = $"Status: Preset saved";
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Save Preset Error", MessageBoxButton.OK, MessageBoxImage.Error);
                StatusText.Text = "Status: Error";
            }
        }

        private void ApplyParametersToUi(AnalysisParameters p)
        {
            // TextBoxes
            MinimumAreaBox.Text = p.minimum_area.ToString();
            AverageAreaBox.Text = p.average_cell_area.ToString();
            ConnectedAreaBox.Text = p.connected_cell_area.ToString();

            LowerIntensityBox.Text = p.lower_intensity.ToString();
            UpperIntensityBox.Text = p.upper_intensity.ToString();

            ScalingBox.Text = p.scaling.ToString(System.Globalization.CultureInfo.InvariantCulture);

            KernelSizeBox.Text = p.kernel_size.ToString();
            OpenIterBox.Text = p.open_iter.ToString();
            CloseIterBox.Text = p.close_iter.ToString();
            ErodeIterBox.Text = p.erode_iter.ToString();
            DilateIterBox.Text = p.dilate_iter.ToString();

            HoleSizeBox.Text = p.hole_size.ToString();
            HoleThresholdBox.Text = p.hole_threshold.ToString();

            // CheckBoxes
            MorphBox.IsChecked = p.morph_checkbox;
            NoiseBox.IsChecked = p.noise;
            OpeningBox.IsChecked = p.opening;
            ClosingBox.IsChecked = p.closing;
            ErodingBox.IsChecked = p.eroding;
            DilatingBox.IsChecked = p.dilating;

            // ComboBox: ImageMethodBox contains ComboBoxItem elements
            SetComboBoxToContent(ImageMethodBox, p.image_method);

            // ComboBox: ContourMethod uses index in your pipeline
            // Ensure index is in range
            if (p.contour_method >= 0 && p.contour_method < ContourMethodBox.Items.Count)
                ContourMethodBox.SelectedIndex = p.contour_method;
            else
                ContourMethodBox.SelectedIndex = 2; // fallback default
        }

        private static void SetComboBoxToContent(System.Windows.Controls.ComboBox combo, string content)
        {
            for (int i = 0; i < combo.Items.Count; i++)
            {
                if (combo.Items[i] is System.Windows.Controls.ComboBoxItem item)
                {
                    if (string.Equals(item.Content?.ToString(), content, StringComparison.OrdinalIgnoreCase))
                    {
                        combo.SelectedIndex = i;
                        return;
                    }
                }
            }

            // If not found, leave current selection alone
        }


        private void LoadPreset_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                var dialog = new OpenFileDialog
                {
                    Title = "Load Parameter Preset",
                    Filter = "CellAnalyzer Preset (*.cellpreset;*.json)|*.cellpreset;*.json"
                };

                if (dialog.ShowDialog() != true)
                    return;

                string json = File.ReadAllText(dialog.FileName);

                var preset = JsonSerializer.Deserialize<ParameterPreset>(json, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

                if (preset == null || preset.parameters == null)
                    throw new Exception("Preset file is invalid or missing parameters.");

                ApplyParametersToUi(preset.parameters);

                StatusText.Text = $"Status: Preset loaded";
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Load Preset Error", MessageBoxButton.OK, MessageBoxImage.Error);
                StatusText.Text = "Status: Error";
            }
        }


        // -------- Parameters --------

        private AnalysisParameters BuildParamsFromUi()
        {
            // Use invariant parsing so decimals work regardless of locale
            var ci = CultureInfo.InvariantCulture;

            int minArea = ParseInt(MinimumAreaBox.Text, "Minimum area");
            int avgArea = ParseInt(AverageAreaBox.Text, "Average cell area");
            int connectedArea = ParseInt(ConnectedAreaBox.Text, "Connected cell area");

            int lower = ParseInt(LowerIntensityBox.Text, "Lower intensity");
            int upper = ParseInt(UpperIntensityBox.Text, "Upper intensity");

            if (lower < 0 || lower > 255) throw new ArgumentException("Lower intensity must be 0–255");
            if (upper < 0 || upper > 255) throw new ArgumentException("Upper intensity must be 0–255");
            if (lower >= upper) throw new ArgumentException("Lower intensity must be less than upper intensity");

            double scaling = ParseDouble(ScalingBox.Text, "Scaling", ci);
            if (scaling <= 0) throw new ArgumentException("Scaling must be > 0");

            bool morph = MorphBox.IsChecked == true;

            // Image method as string from ComboBox
            string imageMethod = (ImageMethodBox.SelectedItem as System.Windows.Controls.ComboBoxItem)?.Content?.ToString() ?? "Sobel";

            // Contour method as numeric (SelectedIndex)
            int contourMethod = ContourMethodBox.SelectedIndex;
            if (contourMethod < 0) contourMethod = 2;

            // Morph-related values (still needed even if morph is off)
            int kernelSize = ParseInt(KernelSizeBox.Text, "Kernel size");
            if (kernelSize < 1) kernelSize = 3;
            if (kernelSize % 2 == 0) throw new ArgumentException("Kernel size must be an odd number");

            int openIter = ParseInt(OpenIterBox.Text, "Open iterations");
            int closeIter = ParseInt(CloseIterBox.Text, "Close iterations");
            int erodeIter = ParseInt(ErodeIterBox.Text, "Erode iterations");
            int dilateIter = ParseInt(DilateIterBox.Text, "Dilate iterations");

            int holeSize = ParseInt(HoleSizeBox.Text, "Hole size");
            int holeThreshold = ParseInt(HoleThresholdBox.Text, "Hole threshold");

            return new AnalysisParameters
            {
                minimum_area = minArea,
                average_cell_area = avgArea,
                connected_cell_area = connectedArea,

                lower_intensity = lower,
                upper_intensity = upper,

                block_size = 100,
                scaling = scaling,

                fluorescence = false,
                fluorescence_scoring = false,

                image_method = imageMethod,
                contour_method = contourMethod,

                morph_checkbox = morph,
                kernel_size = kernelSize,

                noise = NoiseBox.IsChecked == true,
                opening = OpeningBox.IsChecked == true,
                closing = ClosingBox.IsChecked == true,
                eroding = ErodingBox.IsChecked == true,
                dilating = DilatingBox.IsChecked == true,

                open_iter = openIter,
                close_iter = closeIter,
                erode_iter = erodeIter,
                dilate_iter = dilateIter,

                hole_size = holeSize,
                hole_threshold = holeThreshold
            };
        }

        private static int ParseInt(string? text, string fieldName)
        {
            if (!int.TryParse(text, out int value))
                throw new ArgumentException($"{fieldName} must be an integer");
            return value;
        }

        private static double ParseDouble(string? text, string fieldName, CultureInfo ci)
        {
            if (!double.TryParse(text, NumberStyles.Float, ci, out double value))
                throw new ArgumentException($"{fieldName} must be a number");
            return value;
        }

        // -------- Image helper --------

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