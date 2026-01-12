using System;

namespace CellAnalyzer.Desktop.Models
{
    // Wrapper around your AnalysisParameters so you can add metadata later
    public sealed class ParameterPreset
    {
        public int schema_version { get; set; } = 1;

        // Optional: show this in UI later
        public string name { get; set; } = "Preset";

        public DateTime saved_at { get; set; } = DateTime.Now;

        // The actual parameters used by your Python engine
        public AnalysisParameters parameters { get; set; } = new AnalysisParameters();
    }
}
