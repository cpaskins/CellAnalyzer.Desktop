namespace CellAnalyzer.Desktop.Models
{
    public class CellAnalysisResult
    {
        public Counts counts { get; set; }
        public Areas areas { get; set; }
        public Fluorescence fluorescence { get; set; }
        public string overlay_base64 { get; set; }
    }

    public class Counts
    {
        public int cell_count { get; set; }
    }

    public class Areas
    {
        public int total_contour_area { get; set; }
        public double mean_contour_area { get; set; }
        public int total_threshold_area { get; set; }
    }

    public class Fluorescence
    {
        public double[] average_intensities { get; set; }
    }
}
