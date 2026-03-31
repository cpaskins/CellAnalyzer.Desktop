using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;

namespace CellAnalyzer.Desktop.Models
{
    public enum PipelineStatus { Pending, Running, Done, Error }

    public class PipelineItem : INotifyPropertyChanged
    {
        private PipelineStatus _status = PipelineStatus.Pending;
        private int? _cellCount;
        private int? _totalArea;
        private double? _meanArea;
        private string? _overlayPath;
        private string? _errorMessage;

        public int Index { get; set; }
        public string ImagePath { get; set; } = "";
        public string FileName => Path.GetFileName(ImagePath);

        public PipelineStatus Status
        {
            get => _status;
            set { _status = value; OnPropertyChanged(); OnPropertyChanged(nameof(StatusDisplay)); }
        }

        public string StatusDisplay => _status switch
        {
            PipelineStatus.Pending  => "Pending",
            PipelineStatus.Running  => "Running...",
            PipelineStatus.Done     => "Done",
            PipelineStatus.Error    => "Error",
            _                       => "Unknown"
        };

        public int? CellCount
        {
            get => _cellCount;
            set { _cellCount = value; OnPropertyChanged(); OnPropertyChanged(nameof(CellCountDisplay)); }
        }

        public string CellCountDisplay => _cellCount?.ToString() ?? "–";

        public int? TotalArea
        {
            get => _totalArea;
            set { _totalArea = value; OnPropertyChanged(); OnPropertyChanged(nameof(TotalAreaDisplay)); }
        }

        public string TotalAreaDisplay => _totalArea.HasValue ? $"{_totalArea}" : "–";

        public double? MeanArea
        {
            get => _meanArea;
            set { _meanArea = value; OnPropertyChanged(); OnPropertyChanged(nameof(MeanAreaDisplay)); }
        }

        public string MeanAreaDisplay => _meanArea.HasValue ? $"{_meanArea:F2}" : "–";

        public string? OverlayPath
        {
            get => _overlayPath;
            set { _overlayPath = value; OnPropertyChanged(); }
        }

        public string? ErrorMessage
        {
            get => _errorMessage;
            set { _errorMessage = value; OnPropertyChanged(); }
        }

        public string? RunDir { get; set; }

        public event PropertyChangedEventHandler? PropertyChanged;

        private void OnPropertyChanged([CallerMemberName] string? name = null)
            => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }
}
