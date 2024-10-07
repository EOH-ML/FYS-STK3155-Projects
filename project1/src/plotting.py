import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator
import seaborn as sns

def plot_terrain(x, y, z_pred, title=None, filename=None, colorbar=True):
    """
    Plots three dimensional terrain-like functions, using terrain inspired color mapping

    Parameters
    ----------
    x : 2d-array
        meshed array of x-points
    y : 2d-array
        meshed array of y-points
    z : 2d-array
        meshed array of z-points

    title : str
        adds a title to the plot
    filename : str
        if a filename is added, the plot is saved as a PNG to the plots folder
    colorbar : Boolean
        turns the color bar off with False or on with True (default)
    
    Returns
    -------
    Shows the plot (and saves it if a filename is provided)
    """

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif') 

    fig = plt.figure(figsize=(5, 5))  # Justerer størrelsen på figuren (bredde, høyde)
    ax = fig.add_subplot(111, projection='3d')

    z_pred_reshaped = z_pred.reshape(x.shape)
    plot = ax.plot_surface(x, y, z_pred_reshaped, cmap=plt.cm.terrain, linewidth=0, antialiased=True)


    ax.xaxis._axinfo["grid"].update(color='lightgray', linewidth=0.5)  # Hvit/lysegrå, tynn linje for x-aksen
    ax.yaxis._axinfo["grid"].update(color='lightgray', linewidth=0.5)  
    ax.zaxis._axinfo["grid"].update(color='lightgray', linewidth=0.5)  

    # mindre fontstørrelse på aksene og LaTex
    ax.set_xlabel(r'\textbf{x}', fontsize=12)
    ax.set_ylabel(r'\textbf{y}', fontsize=12)
    ax.set_zlabel(r'\textbf{$f$}', fontsize=12)
    
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # color bar
    if colorbar:
        cbar = fig.colorbar(plot, shrink=0.5, aspect=12)  # høyere aspect for å gjøre den smalere
        cbar.ax.tick_params(labelsize=8)  # tallene på fargebaren mindre
    
    ax.set_title(title)

    # lagrer hvis filnavn
    if filename:
        save_file(filename)

def plot_heatmap(heatmap_matrix, x_data, y_data, x_label, y_label, heatbar_label, title, decimals, filename=None):
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
    
    Returns
    -------
    None
        This function does not return any value. It directly displays the heatmap plot.

    """
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=12)

    # plt.figure(figsize=(10, 10))
    min_val_idx = np.unravel_index(np.argmin(heatmap_matrix), heatmap_matrix.shape)
    ax = sns.heatmap(heatmap_matrix, 
                     annot=True, 
                     fmt=f".{decimals}f",
                     linewidths=0.5, 
                     linecolor='white', 
                     xticklabels=x_data, 
                     yticklabels=y_data, 
                     cmap='magma', 
                     cbar=False)#{'label': rf'{heatbar_label}'})
    ax.add_patch(plt.Rectangle((min_val_idx[1], min_val_idx[0]), 1, 1, fill=False, edgecolor='red', lw=3))
    plt.xlabel(rf'{x_label}')
    plt.ylabel(rf'{y_label}')
    plt.title(rf'{title}')
    
    if filename:
        save_file(filename)


def plot_bias_variance_tradeoff(mse, bias, var, filename=None):
    """
    Plots the bias-variance tradeoff, showing the relationships between MSE, bias, and variance
    across different model complexities.

    Parameters:
    -----------
    mse : array-like
        Mean Squared Error (MSE) values at different levels of model complexity.
        
    bias : array-like
        Bias values at different levels of model complexity.
        
    var : array-like
        Variance values at different levels of model complexity.
        
    filename : str, optional
        The name of the file where the plot will be saved. If None (default), the plot will
        be displayed but not saved.

    Returns:
    --------
    None
        Saves the plot
    """

    colors, linestyle = colors_lines()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=12)

    fig, ax1 = plt.subplots(figsize=(6, 8))

    ax1.plot(var, label=r'Variance', color=colors[0], marker='*', linestyle=linestyle[0])
    ax1.set_xlabel(r'Degree of Complexity')
    ax1.set_ylabel(r'Variance')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create a second y-axis for bias and mse
    ax2 = ax1.twinx()
    ax2.plot(bias, linestyle=linestyle[1], label=r'Bias', color=colors[1], marker='*')
    ax2.plot(mse, linestyle=linestyle[2], label=r'MSE', color=colors[2], marker='o')
    ax2.set_ylabel(r'Bias / MSE')
    ax2.tick_params(axis='y', labelcolor='black')
    lines = ax1.get_lines() + ax2.get_lines()  # Get lines from both axes
    labels = [line.get_label() for line in lines]  # Get corresponding labels
    ax1.legend(lines, labels, loc='upper center')  # Adjust ncol for layout

    fig.tight_layout()  # To ensure there's no overlap

    if filename:
        save_file(filename)

def plot_betas_with_ci(betas, color, degree, ci, filename=None):
    """
    Plots the estimated regression coefficients (betas) along with their 95% confidence intervals (CI)
    for a polynomial model of a given degree.

    Parameters:
    -----------
    betas : array-like
        A list or array of estimated coefficients (betas) for the model.
        
    color : str
        Color to be used for plotting both the betas and the confidence intervals (CIs).
        
    degree : int
        Degree of the polynomial model. This is included in the plot title and label for context.
        
    ci : array-like
        Confidence intervals (CIs) for each beta value, provided as an array or list of error margins.
        This is used to draw error bars around the betas.
        
    filename : str, optional
        The name of the file where the plot will be saved. If None (default), the plot will
        be displayed but not saved.

    Returns:
    --------
    None
        The function will either display the plot or save it to a file (if `filename` is provided).
    """
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=12)
    # Plot betas with lines and error bars
    plt.plot(np.arange(len(betas)), betas, marker='.', linestyle='-', markersize=8, 
            color=color, label=rf'$\beta$ for deg: {degree}')
    plt.errorbar(np.arange(len(betas)), betas, yerr=ci, ecolor=color, 
                capsize=5, elinewidth=2, alpha=0.5, fmt='none')  # fmt='none' hides individual points
    plt.xlabel(r'Index of $\mathbf{\beta}$')
    plt.ylabel(r'$\mathbf{\beta}$ values')
    # plt.ylim(-20, 20)
    plt.title(rf'Coefficients with 95 percent CI for degree {degree}')
    plt.legend()
    plt.tight_layout()  # Adjust layout for better readability
    if filename:
        save_file(filename)

def plot_1d(*fs, x_values, x_label, y_label, title, log_plot=False, filename=None):
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

    Returns:
    --------
    None
        The function either displays the plot or saves it to a file if `filename` is provided.
    """
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=12)
    colors, line_styles = colors_lines()

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
        save_file(filename)

def save_file(filename):
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
    plt.savefig(f'./figures/{filename}', format='png', dpi=300, bbox_inches='tight')

def colors_lines():
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
    colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928']  # Blue, Green, Red, Orange, Purple, Brown
    line_styles = ['-', '--', '-.', ':']
    return colors, line_styles