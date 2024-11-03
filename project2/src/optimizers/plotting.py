import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Plotting:

    def __init__(self):
        pass

    def plot_1d(self, *fs, x_values, x_label, y_label, title, log_plot=False, filename=None, is_minimal=False):
        """
        Plots multiple 1D functions (or data series) on the same set of x-values with options 
        for regular or logarithmic scaling.

        Parameters:
        -----------
        *fs : list of tuples
            Each tuple should contain two elements: the first element is an array-like object 
            representing the y-values (function values or data points), and the second element 
            is a string that serves as the label for the plot legend.
            
        x_values : array-like
            An array or list representing the x-values for the plot (the domain over which the 
            functions are evaluated).
            
        x_label : str
            Label for the x-axis.
            
        y_label : str
            Label for the y-axis.
            
        title : str
            Title of the plot. If None, no title will be displayed.
            
        log_plot : bool, optional
            If True, the plot will use a logarithmic scale for the x-axis (default is False).
            
        filename : str, optional
            If provided, the plot will be saved as a PNG file with the specified filename. 
            If None (default), the plot will be displayed but not saved.

        is_minimal : bool, optional
            For publication sizd plots

        Returns:
        --------
        None
            The function either displays the plot or saves it to a file.
        """

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        ### Håkons insert
        if is_minimal:
            plt.figure(figsize=(5, 5))
            plt.rc('font', size=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            # Fewer ticks on the axes
            plt.locator_params(axis='x', nbins=5)
            plt.locator_params(axis='y', nbins=5)
        else:
            plt.figure()
            plt.rc('font', size=12)
        ### end of Håkons insert

        colors, line_styles = self._colors_lines()

        if title:
            plt.title(rf'{title}')

        if log_plot:
            for i, f in enumerate(fs):
                plt.semilogx(x_values, f[0], label=rf'{f[1]}', linestyle=line_styles[i % len(line_styles)], color=colors[i % len(colors)])

        else:
            for i, f in enumerate(fs):
                plt.plot(x_values, f[0], label=rf'{f[1]}', linestyle=line_styles[i % len(line_styles)], color=colors[i % len(colors)])
        

        plt.xlabel(rf'{x_label}')
        plt.ylabel(rf'{y_label}')

        plt.legend()
        if filename:
            self._save_file(filename)

    def heatmap(self, heatmap_matrix, x_data, y_data, x_label, y_label, title, decimals, filename=None, is_minimal=False):
        """
        Plots heatmap given a matrix with data, and highlights the smallest value

        Parameters
        ----------
        heatmap_matrix : 2D array
            A matrix containing the values to be represented in the heatmap.

        x_data : 1d-array
            The labels for the x-axis, corresponding to the columns of the heatmap_matrix.

        y_data : 1d-array 
            The labels for the y-axis, corresponding to the rows of the heatmap_matrix.

        x_label : str
            The label for the x-axis.

        y_label : str
            The label for the y-axis.

        heatbar_label : str
            The label for the color bar associated with the heatmap.

        is_minimal : bool, optional
            For publication sizd plots
        
        Returns
        -------
        None
            This function does not return any value. It directly displays (and/or saves) the heatmap plot.

        """
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        ### Håkons insert
        if is_minimal:
            plt.figure(figsize=(5, 5))
            plt.rc('font', size=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            # Fewer ticks on the axes
            plt.locator_params(axis='x', nbins=5)
            plt.locator_params(axis='y', nbins=5)
        else:
            plt.figure()
            plt.rc('font', size=12)
        ### end of Håkons insert

        min_val_idx = np.unravel_index(np.argmin(heatmap_matrix), heatmap_matrix.shape)
        ax = sns.heatmap(heatmap_matrix, 
                        annot=True, 
                        fmt=f".{decimals}f",
                        linewidths=0.5, 
                        linecolor='white', 
                        xticklabels=x_data, 
                        yticklabels=y_data, 
                        cmap=sns.diverging_palette(220,20,as_cmap=True), 
                        cbar=False)
        ax.add_patch(plt.Rectangle((min_val_idx[1], min_val_idx[0]), 1, 1, fill=False, edgecolor='red', lw=3))
        plt.xlabel(rf'{x_label}')
        plt.ylabel(rf'{y_label}')
        plt.title(rf'{title}')
        
        if filename:
            self._save_file(filename)

    def _save_file(self, filename):
        """
        Saves the current plot to a PNG file with the specified filename in the "figures" directory.

        Parameters:
        -----------
        filename : str
            The name of the file (with extension) to which the plot will be saved.
            The file is saved in the "figures" folder, which should exist or be created beforehand.
        
        Returns:
        --------
        None
            This function saves the file and does not return any value.
        """
        plt.savefig(f'{filename}', format='png', dpi=300, bbox_inches='tight')

    def _colors_lines(self):
        """
        Returns predefined sets of colors and line styles for consistent plotting aesthetics.

        Returns:
        --------
        colors : list of str
            A list of hex color codes representing six different colors: Blue, Green, Red, 
            Orange, Purple, and Brown.
            
        line_styles : list of str
            A list of line style patterns for plotting: solid ('-'), dashed ('--'), dash-dot ('-.'),
            and dotted (':').
        """
        colors = plt.get_cmap("Dark2").colors
        line_styles = ['-', '--', '-.', ':']
        return colors, line_styles
