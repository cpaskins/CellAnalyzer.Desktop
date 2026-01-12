// Strongly typed params so serialization is stable
public sealed class AnalysisParameters
{
    public int minimum_area { get; set; }
    public int average_cell_area { get; set; }
    public int connected_cell_area { get; set; }

    public int lower_intensity { get; set; }
    public int upper_intensity { get; set; }

    public int block_size { get; set; }
    public double scaling { get; set; }

    public bool fluorescence { get; set; }
    public bool fluorescence_scoring { get; set; }

    public string image_method { get; set; } = "Sobel";
    public int contour_method { get; set; }

    public bool morph_checkbox { get; set; }
    public int kernel_size { get; set; }

    public bool noise { get; set; }
    public bool opening { get; set; }
    public bool closing { get; set; }
    public bool eroding { get; set; }
    public bool dilating { get; set; }

    public int open_iter { get; set; }
    public int close_iter { get; set; }
    public int erode_iter { get; set; }
    public int dilate_iter { get; set; }

    public int hole_size { get; set; }
    public int hole_threshold { get; set; }
}

