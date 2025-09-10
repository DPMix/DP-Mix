# -*- coding: utf-8 -*-
"""
To plot the figures.
"""
import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, X, Y, Descriptions, X_label, Y_label, name,Titles = None, condition=False,shadow_margin=0.05):
        self.X = X
        self.Y = Y
        self.Desc = Descriptions
        self.XL = X_label
        self.YL = Y_label
        self.name = name
        self.condition = condition
        self.markers = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'h', '*']  # Professional markers
        self.Line_style = ['-', '--', '-.', ':']  # Clean line styles
        self.colors = ['blue','red','darkblue','green','fuchsia','indigo','teal','lime','blue','black','orange','violet','lightblue']
        self.shadow_margin = shadow_margin
        self.Titles = Titles
        self.LEN = len(Y[0])
        self.loc1 = 'lower right'
        self.loc2 = 'lower left'        
        self.loc3 = 'upper right'
        self.loc4 = 'upper left'         
    def simple_plot(self, y_max=None, x_tight=False,xx = 0):
        if int(xx)==0:
            loc_1 = self.loc2
        elif int(xx)==1:
            loc_1 = self.loc1
        elif int(xx)==3:
            loc_1 = self.loc4
        elif int(xx)==2:
            loc_1 = self.loc3
            

            
        """
        Plots multiple Y datasets against a single X dataset with unique styles.
        Designed for high-quality publication-ready output.

        Parameters:
        - y_max: Optional float. Maximum value for the y-axis. Defaults to max of Y data.
        - x_tight: Optional boolean. If True, tightens the x-axis range to minimize empty space.
        """
        # Create a rectangular figure
        plt.figure(figsize=(10, 6))  # Adjusted for rectangular aspect ratio

        # Loop through Y datasets and plot each
        for i, y in enumerate(self.Y):
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            line_style = self.Line_style[i % len(self.Line_style)]
            plt.plot(
                self.X, y, color=color, linestyle=line_style,
                marker=marker, markersize=12,  # Increased marker size
                markerfacecolor='none',  # Hollow markers
                markeredgewidth=4, markeredgecolor=color,  # Thicker marker edges
                linewidth=4,  # Thick line width
                label=self.Desc[i]
            )

        # Add labels and legend
        plt.xlabel(self.XL, fontsize=30, fontweight='bold')  # Font size for x-axis
        plt.ylabel(self.YL, fontsize=30, fontweight='bold')  # Font size for y-axis
        plt.legend(fontsize=20, loc=loc_1, frameon=True, framealpha=0.9, edgecolor='gray')  # Legend formatting
        plt.grid(linestyle='--', alpha=0.7, linewidth=0.8)  # Subtle gridlines for better readability

        # Set y-axis limit
        if y_max is None:
            y_max = max([max(y) for y in self.Y])
        plt.ylim(0, y_max)

        # Adjust x-axis range if x_tight is True
        if x_tight:
            plt.xlim(min(self.X) - 0.1, max(self.X) + 0.1)

        # Refine ticks: Ensure at most 5 ticks on both axes
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))

        # Improve ticks appearance
        plt.xticks(fontsize=25, fontweight='bold')  # Tick font size and style
        plt.yticks(fontsize=25, fontweight='bold')  # Tick font size and style

        # Save the plot
        plt.tight_layout()
        plt.savefig(self.name, format='png', dpi=600)  # Higher DPI for better quality
        plt.show()


    
    def simple_plot20(self, y_max=None, x_tight=False, xx=0, inset_zoom=False):
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
        if int(xx) == 0:
            loc_1 = self.loc2
        elif int(xx) == 1:
            loc_1 = self.loc1
        elif int(xx) == 3:
            loc_1 = self.loc4
        elif int(xx) == 2:
            loc_1 = self.loc3
    
        fig, ax = plt.subplots(figsize=(10, 6))
        loc_1 = 'upper center'
        # Plot main curves
        for i, y in enumerate(self.Y):
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            line_style = self.Line_style[i % len(self.Line_style)]
            ax.plot(
                self.X, y, color=color, linestyle=line_style,
                marker=marker, markersize=12,
                markerfacecolor='none',
                markeredgewidth=4, markeredgecolor=color,
                linewidth=4,
                label=self.Desc[i]
            )
    
        ax.set_xlabel(self.XL, fontsize=30, fontweight='bold')
        ax.set_ylabel(self.YL, fontsize=30, fontweight='bold')
        legend = ax.legend(
    fontsize=20,
    loc='center',                          # Position relative to bbox
    bbox_to_anchor=(0.3, 0.8),             # X, Y (fraction of axes)
    frameon=True, framealpha=0.9,
    edgecolor='gray'
)

        
        ax.grid(linestyle='--', alpha=0.7, linewidth=0.8)
    
        if y_max is None:
            y_max = max([max(y) for y in self.Y])
        ax.set_ylim(0, y_max)
    
        if x_tight:
            ax.set_xlim(min(self.X) - 0.1, max(self.X) + 0.1)
    
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(axis='both', labelsize=25)
        # Improve ticks appearance
        plt.xticks(fontsize=25, fontweight='bold')  # Tick font size and style
        plt.yticks(fontsize=25, fontweight='bold')  # Tick font size and style
        
        # Zoomed inset
        if inset_zoom:
            # Create inset axes
            axins = inset_axes(ax, width="35%", height="35%", loc='upper right')
    
            for i, y in enumerate(self.Y):
                color = self.colors[i % len(self.colors)]
                marker = self.markers[i % len(self.markers)]
                line_style = self.Line_style[i % len(self.Line_style)]
                axins.plot(
                    self.X, y, color=color, linestyle=line_style,
                    marker=marker, markersize=6,
                    markerfacecolor='none',
                    markeredgewidth=2, markeredgecolor=color,
                    linewidth=2
                )
    
            # Set zoom range (x ≥ 8)
            zoom_start = 3
            x_zoom = [x for x in self.X if x >= zoom_start]
            if len(x_zoom) >= 2:
                x1, x2 = x_zoom[0], x_zoom[-1]
                axins.set_xlim(x1, x2)
                y_vals = [y[ix] for y in self.Y for ix, x in enumerate(self.X) if x1 <= x <= x2]
                axins.set_ylim(min(y_vals), max(y_vals))
                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    
            axins.tick_params(axis='both', labelsize=12)
    
    
            # Improve ticks appearance
        plt.xticks(fontsize=12, fontweight='bold')  # Tick font size and style
        plt.yticks(fontsize=12, fontweight='bold')  # Tick font size and style

        plt.tight_layout()
        plt.savefig(self.name, format='png', dpi=600)
        plt.show()
    
    



    
    
    






















    def simple_plot_(self, y_max=None, x_tight=False,xx = False):
        if not xx:
            
            loc_1 = 'upper right'
        else:
            loc_1 = 'upper left'
            
        """
        Plots multiple Y datasets against a single X dataset with unique styles.
        Designed for high-quality publication-ready output.

        Parameters:
        - y_max: Optional float. Maximum value for the y-axis. Defaults to max of Y data.
        - x_tight: Optional boolean. If True, tightens the x-axis range to minimize empty space.
        """
        # Create a rectangular figure
        plt.figure(figsize=(10, 6))  # Adjusted for rectangular aspect ratio

        # Loop through Y datasets and plot each
        for i, y in enumerate(self.Y):
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            line_style = self.Line_style[i % len(self.Line_style)]
            plt.plot(
                self.X, y, color=color, linestyle=line_style,  # Increased marker size
                markerfacecolor='none',  # Hollow markers
                markeredgewidth=3, markeredgecolor=color,  # Thicker marker edges
                linewidth=3,  # Thick line width
                label=self.Desc[i]
            )

        # Add labels and legend
        plt.xlabel(self.XL, fontsize=30, fontweight='bold')  # Font size for x-axis
        plt.ylabel(self.YL, fontsize=30, fontweight='bold')  # Font size for y-axis
        plt.legend(fontsize=20, loc=loc_1, frameon=True, framealpha=0.9, edgecolor='gray')  # Legend formatting
        plt.grid(linestyle='--', alpha=0.7, linewidth=0.8)  # Subtle gridlines for better readability

        # Set y-axis limit
        if y_max is None:
            y_max = max([max(y) for y in self.Y])
        plt.ylim(0, y_max)

        # Adjust x-axis range if x_tight is True
        if x_tight:
            plt.xlim(min(self.X) - 0.1, max(self.X) + 0.1)

        # Refine ticks: Ensure at most 5 ticks on both axes
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))

        # Improve ticks appearance
        plt.xticks(fontsize=25, fontweight='bold')  # Tick font size and style
        plt.yticks(fontsize=25, fontweight='bold')  # Tick font size and style

        # Save the plot
        plt.tight_layout()
        plt.savefig(self.name, format='png', dpi=600)  # Higher DPI for better quality
        plt.show()

    def merged_plot(self, Y_1max=None, Y_2max=None):
        c_color1 = ['blue','red']
        c_color2 = ['green','black']
        """
        Merged plot with two y-axes: Y_1 on the left, Y_2 on the right.

        Parameters:
        - Y_1max: Optional float. Maximum value for the left y-axis (Y_1).
        - Y_2max: Optional float. Maximum value for the right y-axis (Y_2).
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))  # Rectangular aspect ratio

        # Plot Y_1 on the left y-axis
        for i, y in enumerate(self.Y[0]):  # self.Y[0] corresponds to Y_1 = [y1, y2, ...]
            color = c_color2[i%2]
            marker = self.markers[i % len(self.markers)]
            line_style = self.Line_style[i%2]  # Solid line for Y_1
            ax1.plot(
                self.X, y, color=color, linestyle=line_style,
                marker=marker, markersize=16,  # Increased marker size
                markerfacecolor='none',  # Hollow markers
                markeredgewidth=3, markeredgecolor=color, linewidth=3,
                label=self.Desc[0][i]
            )
        ax1.set_xlabel(self.XL, fontsize=30, fontweight='bold')
        ax1.set_ylabel(self.YL[0], fontsize=30, fontweight='bold', color='black')  # Left y-label
        ax1.tick_params(axis='y', labelsize=25, colors='black')
        ax1.tick_params(axis='x', labelsize=25)

        # Set maximum range for Y_1
        if Y_1max is not None:
            ax1.set_ylim(0, Y_1max)
        left_legend = ax1.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.9, edgecolor='gray')
        ax1.add_artist(left_legend)  # Keep the left legend independent
        ax1.text(0.05, 0.85, 'data1', fontsize=12, ha='left', transform=ax1.transAxes)  # Small label below left legend

        # Create second y-axis for Y_2
        ax2 = ax1.twinx()
        for i, z in enumerate(self.Y[1]):  # self.Y[1] corresponds to Y_2 = [z1, z2, ...]
            color = c_color1[i%2]
            marker = self.markers[i % len(self.markers)]
            line_style = self.Line_style[i%2]  # Dashed line for Y_2
            ax2.plot(
                self.X, z, color=color, linestyle=line_style,
                marker=marker, markersize=16,  # Increased marker size
                markerfacecolor='none',  # Hollow markers
                markeredgewidth=3, markeredgecolor=color, linewidth=3,
                label=self.Desc[1][i]
            )
        ax2.set_ylabel(self.YL[1], fontsize=30, fontweight='bold', color='black')  # Right y-label
        ax2.tick_params(axis='y', labelsize=25, colors='black')

        # Set maximum range for Y_2
        if Y_2max is not None:
            ax2.set_ylim(0, Y_2max)
        right_legend = ax2.legend(fontsize=12, loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')
        ax2.text(0.95, 0.85, 'data2', fontsize=12, ha='right', transform=ax2.transAxes)  # Small label below right legend

        # Add gridlines and adjust layout
        ax1.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.name, format='png', dpi=600)  # High-quality save
        plt.show()


    def cdf_plot(self):
        """
        Plots the first, middle, and last curves in self.Y and creates a shaded region
        around them with a gray shadow effect.
        """
        plt.figure(figsize=(10, 6))  # Rectangular figure for better layout

        # Plot the first curve (Y1)
        plt.plot(
            self.X, self.Y[0], color=self.colors[0], linestyle=self.Line_style[0],
            linewidth=2.5, label=self.Desc[0]
        )

        # Plot the middle curve (Ym)
        middle_idx = len(self.Y) // 2
        plt.plot(
            self.X, self.Y[middle_idx], color=self.colors[1], linestyle=self.Line_style[1],
            linewidth=3, label=self.Desc[middle_idx]
        )

        # Plot the last curve (Yn)
        plt.plot(
            self.X, self.Y[-1], color=self.colors[2], linestyle=self.Line_style[2],
            linewidth=3, label=self.Desc[-1]
        )

        # Create a shaded region around the first and last curves
        y1 = np.array(self.Y[0])
        y_last = np.array(self.Y[-1])
        shadow_margin = 0.05 * (np.max(self.Y) - np.min(self.Y))  # Extend shadow slightly above and below
        plt.fill_between(
            self.X, y1 - shadow_margin, y_last + shadow_margin, color='gray', alpha=0.35,
            label=r'$0<\alpha<1$'
        )

        # Add axis labels
        plt.xlabel(self.XL, fontsize=30, fontweight='bold')
        plt.ylabel(self.YL, fontsize=30, fontweight='bold')

        # Increase tick size
        plt.xticks(fontsize=25, fontweight='bold')
        plt.yticks(fontsize=25, fontweight='bold')

        # Add legend
        plt.legend(fontsize=18, loc='upper left', frameon=True, framealpha=0.9, edgecolor='gray')

        # Add grid and tighten layout
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the plot
        plt.savefig(self.name, format='png', dpi=600)  # High-quality save
        plt.show()


    def extended_cdf_plot(self, tick_step=None):
        """
        Plots the elements of self.Y (Y_1, Y_2, Y_3, Y_4) in a 2x2 subplot grid
        with shadows for the first and last curves in each subplot.
        Adds a general x-label, a shared y-label, and unique titles for each subplot.
        Includes legends and an option to increase the number of ticks on the axes.
        """
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))  # 2x2 grid of subplots
        axes = axes.flatten()  # Flatten for easier indexing

        for i, (Y_i, ax) in enumerate(zip(self.Y, axes)):
            # Plot the first curve of Y_i
            ax.plot(
                self.X, Y_i[0], color=self.colors[0], linestyle=self.Line_style[0],
                linewidth=3, label=self.Desc[0] 
            )

            # Plot the last curve of Y_i
            ax.plot(
                self.X, Y_i[-1], color=self.colors[1], linestyle=self.Line_style[1],
                linewidth=3, label=self.Desc[self.LEN-1] 
            )

            # Create a shaded region between the first and last curves
            y1 = np.array(Y_i[0], dtype=float)
            y_last = np.array(Y_i[-1], dtype=float)
            ax.fill_between(
                self.X, y1 - self.shadow_margin, y_last + self.shadow_margin,
                color='gray', alpha=0.3
            )

            # Set subplot title
            ax.set_title(self.Titles[i], fontsize=25, fontweight='bold')

            # Add grid and ticks
            ax.grid(linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', which='major', labelsize=20)

            # Adjust ticks if tick_step is provided
            if tick_step:
                ax.set_xticks(np.arange(self.X.min(), self.X.max() + tick_step, tick_step))
                ax.set_yticks(np.arange(y1.min(), y_last.max() + tick_step, tick_step))

            # Add legend
            ax.legend(fontsize=20, loc='lower right', frameon=True, framealpha=0.6, edgecolor='gray')

        # Optional y-label for all plots (can be removed if not needed)
        fig.text(0.04, 0.5, self.YL, va='center', rotation='vertical', fontsize=25, fontweight='bold')

        # General x-label for the entire figure
        fig.text(0.5, 0.02, self.XL, ha='center', fontsize=22, fontweight='bold')

        # Adjust layout
        plt.tight_layout(rect=[0.05, 0.05, 1, 1])  # Leave space for shared labels
        plt.savefig(self.name, format='png', dpi=600)  # High-quality save
        plt.show()
  
    def box_plot(self, y_max=None):
        """
        Creates a grouped box plot where each group is associated with a specific X value,
        and each group contains box plots for Y1i, Y2i, ...
        """
        plt.figure(figsize=(10, 5))

        num_groups = len(self.X)  # Number of x-axis positions
        num_categories = len(self.Y)  # Number of categories (Y1, Y2, ...)
        group_width = 0.5  # Total width for each group on the x-axis
        category_width = group_width / num_categories  # Width for each boxplot within a group

        for i in range(len(self.X)):  # Iterate over each x position
            positions = [self.X[i]*2 + j * category_width*4 for j in range(num_categories)]
            data = [self.Y[j][i] for j in range(num_categories)]  # Get all Yij values for x_i
            
            for j in range(num_categories):
                # Assign different colors for Y1 and Y2 elements
                edge_color = self.colors[j % len(self.colors)]
                
                # Create box plots for each category at each X position
                plt.boxplot(data[j], positions=[positions[j]], widths=category_width * 3,
                            patch_artist=True,  # Enable custom colors
                            boxprops=dict(facecolor='white', edgecolor=edge_color, linewidth=3),
                            medianprops=dict(color='black', linewidth=3),
                            whiskerprops=dict(color=edge_color, linewidth=3),
                            capprops=dict(color=edge_color, linewidth=3),
                            flierprops=dict(marker='o', color='gray', alpha=0.1))

        # Set x-axis and y-axis labels
        plt.xlabel(self.XL, fontsize=30, fontweight='bold')
        plt.ylabel(self.YL, fontsize=30, fontweight='bold')

        # Add a legend for each category
        legend_elements = [
            plt.Line2D([0], [0], color=self.colors[i % len(self.colors)], lw=4, label=desc)
            for i, desc in enumerate(self.Desc)
        ]
        plt.legend(handles=legend_elements, fontsize=20, loc='lower left', frameon=True)

        # Set x-axis tick labels
        plt.xticks([self.X[i]*2.02 +0.5+ (num_categories - 1) * category_width / 2 for i in range(len(self.X))],
                   [x for x in ((self.X))], fontsize=25)


        # Set y-axis limit if y_max is provided
        if y_max is not None:
            plt.ylim(0,y_max)
        plt.tick_params(axis='y', labelsize=25)

        # Add grid
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the plot
        plt.tight_layout()
        plt.savefig(self.name, format='png', dpi=600)
        plt.show()


    def box_plot_(self, y_max=None):
        """
        Creates a grouped box plot where each group is associated with a specific X value,
        and each group contains box plots for Y1i, Y2i, ...
        """
        plt.figure(figsize=(10, 5))

        num_groups = len(self.X)  # Number of x-axis positions
        num_categories = len(self.Y)  # Number of categories (Y1, Y2, ...)
        group_width = 0.3  # Total width for each group on the x-axis
        category_width = group_width / num_categories  # Width for each boxplot within a group

        for i in range(len(self.X)):  # Iterate over each x position
            positions = [self.X[i]*2 + j * category_width for j in range(num_categories)]
            data = [self.Y[j][i] for j in range(num_categories)]  # Get all Yij values for x_i
            
            for j in range(num_categories):
                # Assign different colors for Y1 and Y2 elements
                edge_color = self.colors[j % len(self.colors)]
                
                # Create box plots for each category at each X position
                plt.boxplot(data[j], positions=[positions[j]], widths=category_width * 0.8,
                            patch_artist=True,  # Enable custom colors
                            boxprops=dict(facecolor='white', edgecolor=edge_color, linewidth=3),
                            medianprops=dict(color='black', linewidth=3),
                            whiskerprops=dict(color=edge_color, linewidth=3),
                            capprops=dict(color=edge_color, linewidth=3),
                            flierprops=dict(marker='o', color='gray', alpha=0.25))

        # Set x-axis and y-axis labels
        plt.xlabel(self.XL, fontsize=30, fontweight='bold')
        plt.ylabel(self.YL, fontsize=30, fontweight='bold')

        # Add a legend for each category
        legend_elements = [
            plt.Line2D([0], [0], color=self.colors[i % len(self.colors)], lw=4, label=desc)
            for i, desc in enumerate(self.Desc)
        ]
        plt.legend(handles=legend_elements, fontsize=20, loc='lower right', frameon=True)

        # Set x-axis tick labels
        plt.xticks([self.X[i]*2.02 + (num_categories - 1) * category_width / 2 for i in range(len(self.X))],
                   [x for x in ((self.X))], fontsize=25)


        # Set y-axis limit if y_max is provided
        if y_max is not None:
            plt.ylim(0,y_max)
        plt.tick_params(axis='y', labelsize=25)

        # Add grid
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the plot
        plt.tight_layout()
        plt.savefig(self.name, format='png', dpi=600)
        plt.show()

    def violin_plot(self, y_max=None):
        """
        Creates a grouped violin plot (vase plot) where each group is associated with a specific X value,
        and each group contains violins for Y1i, Y2i, ...
        """
        plt.figure(figsize=(10, 5))
    
        num_groups = len(self.X)  # Number of x-axis positions
        num_categories = len(self.Y)  # Number of categories (Y1, Y2, ...)
        group_width = 0.3  # Total width for each group on the x-axis
        category_width = group_width / num_categories  # Width for each violin within a group
    
        for i in range(len(self.X)):  # Iterate over each x position
            positions = [self.X[i]*2 + j * category_width for j in range(num_categories)]
            data = [self.Y[j][i] for j in range(num_categories)]  # Yij data per group
    
            for j in range(num_categories):
                violin_parts = plt.violinplot(dataset=data[j],
                                              positions=[positions[j]],
                                              widths=category_width * 0.8,
                                              showmedians=True,
                                              showextrema=False)
    
                color = self.colors[j % len(self.colors)]
    
                for pc in violin_parts['bodies']:
                    pc.set_facecolor('white')
                    pc.set_edgecolor(color)
                    pc.set_linewidth(2)
                    pc.set_alpha(1)
    
                if 'cmedians' in violin_parts:
                    violin_parts['cmedians'].set_color('black')
                    violin_parts['cmedians'].set_linewidth(3)
    
        # Labels
        plt.xlabel(self.XL, fontsize=30, fontweight='bold')
        plt.ylabel(self.YL, fontsize=30, fontweight='bold')
    
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color=self.colors[i % len(self.colors)], lw=4, label=desc)
            for i, desc in enumerate(self.Desc)
        ]
        plt.legend(handles=legend_elements, fontsize=20, loc='upper right', frameon=True)
    
        # X-ticks
        plt.xticks([self.X[i]*2.02 + (num_categories - 1) * category_width / 2 for i in range(len(self.X))],
                   [x for x in self.X], fontsize=25)
    
        # Y-ticks
        if y_max is not None:
            plt.ylim(0, y_max)
        plt.tick_params(axis='y', labelsize=25)
    
        # Grid and layout
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
    
        # Save and show
        plt.savefig(self.name, format='png', dpi=600)
        plt.show()


































'''
X = [1, 2, 3]  # x-axis positions
Y1 = [[1.2, 1.5, 1.8], [1.0, 1.3, 1.7], [1.4, 1.6, 1.9], [2.4, 2.6, 2.9]]  # Box plots for Y1
Y2 = [[2.2, 2.5, 2.8], [2.0, 2.3, 2.7], [2.4, 2.6, 2.9], [2.4, 2.6, 2.9]]  # Box plots for Y2
Y3 = [[2.2, 2.5, 2.8], [2.0, 2.3, 2.7], [2.4, 2.6, 2.9], [2.4, 2.6, 2.9]]
Y = [Y1, Y2,Y3]
Descriptions = ['Category 1', 'Category 2','3']
X_label = 'X Label'
Y_label = 'Y Label'
name = 'violin_plot.png'

plotter = Plotter(X, Y, Descriptions, X_label, Y_label, name)
plotter.violin_plot(y_max=4)

'''








'''
X = [1, 2, 3]  # x-axis positions
Y1 = [[1.2, 1.5, 1.8], [1.0, 1.3, 1.7], [1.4, 1.6, 1.9], [2.4, 2.6, 2.9]]  # Box plots for Y1
Y2 = [[2.2, 2.5, 2.8], [2.0, 2.3, 2.7], [2.4, 2.6, 2.9], [2.4, 2.6, 2.9]]  # Box plots for Y2
Y3 = [[2.2, 2.5, 2.8], [2.0, 2.3, 2.7], [2.4, 2.6, 2.9], [2.4, 2.6, 2.9]]
Y = [Y1, Y2,Y3]
Descriptions = ['Category 1', 'Category 2','3']
X_label = 'X Label'
Y_label = 'Y Label'
name = 'box_plot.png'

plotter = Plotter(X, Y, Descriptions, X_label, Y_label, name)
plotter.box_plot_(y_max=4)
'''


# Example usage:
'''
X = [1, 2, 3]  # x-axis positions
Y1 = [[1.2, 1.5, 1.8], [1.0, 1.3, 1.7], [1.4, 1.6, 1.9]]  # Box plots for Y1
Y2 = [[2.2, 2.5, 2.8], [2.0, 2.3, 2.7], [2.4, 2.6, 2.9]]  # Box plots for Y2

Y = [Y1, Y2]
Descriptions = ['Category 1', 'Category 2']
X_label = 'X Label'
Y_label = 'Y Label'
name = 'box_plot.png'

plotter = Plotter(X, Y, Descriptions, X_label, Y_label, name)
plotter.box_plot(y_max=4)
'''
        
        
        

# Example usage:
''' 
X = [1, 2, 3]  # x-axis positions
Y1 = [[1.2, 1.5, 1.8], [1.0, 1.3, 1.7], [1.4, 1.6, 1.9]]  # Box plots for Y1
Y2 = [[2.2, 2.5, 2.8], [2.0, 2.3, 2.7], [2.4, 2.6, 2.9]]  # Box plots for Y2
Y3 = [[3.2, 3.5, 3.8], [3.0, 3.3, 3.7], [3.4, 3.6, 3.9]]  # Box plots for Y3

Y = [Y1, Y2, Y3]
Descriptions = ['Category 1', 'Category 2', 'Category 3']
X_label = 'X Label'
Y_label = 'Y Label'
name = 'box_plot.png'

plotter = Plotter(X, Y, Descriptions, X_label, Y_label, name)
plotter.box_plot()
'''
        
        
        
        
        
        
        
        
        
        
        
# Example usage:
'''
X = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
Y_1 = [[0.1, 0.3, 0.5, 0.7, 0.9, 1.0], [0.15, 0.35, 0.55, 0.75, 0.95, 1.0], [0.2, 0.4, 0.6, 0.8, 0.95, 1.0]]
Y_2 = [[0.05, 0.25, 0.45, 0.65, 0.85, 1.0], [0.1, 0.3, 0.5, 0.7, 0.9, 1.0], [0.15, 0.35, 0.55, 0.75, 0.95, 1.0]]
Y_3 = [[0.2, 0.4, 0.6, 0.8, 1.0, 1.1], [0.25, 0.45, 0.65, 0.85, 1.0, 1.05], [0.3, 0.5, 0.7, 0.9, 1.1, 1.2]]
Y_4 = [[0.1, 0.3, 0.5, 0.7, 0.9, 1.0], [0.2, 0.4, 0.6, 0.8, 1.0, 1.1], [0.3, 0.5, 0.7, 0.9, 1.1, 1.2]]
Y = [Y_1, Y_2, Y_3, Y_4]
Descriptions = ['1','2','3']
Titles = ['Set 1', 'Set 2', 'Set 3', 'Set 4']
X_label = 'Common X Label'
Y_label = 'Common Y Label'
plotter = Plotter(X, Y, Descriptions,  X_label, Y_label, 'extended_cdf_plot_with_titles.png',Titles, shadow_margin=0.2)
plotter.extended_cdf_plot()

'''




'''
# Example usage:
X = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
Y = [
   [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    [0.05, 0.25, 0.45, 0.65, 0.85, 1.0],
   [0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
 ]
Descriptions = ['τ = 0.1', 'τ = 0.5', 'τ = 1.0']
plotter = Plotter(X, Y, Descriptions, 'X Label', 'Y Label', 'cdf_plot.png')
plotter.cdf_plot()
'''

'''
# Example usage:
X = [1, 2, 3, 4, 5]
Y_1 = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
Y_2 = [[5, 4, 3, 2, 1], [6, 5, 4, 3, 2], [7, 6, 5, 4, 3]]
Descriptions = [['Dataset 1', 'Dataset 2', 'Dataset 3'],['Dataset 1', 'Dataset 2', 'Dataset 3']]
YL = ['Y Label 1', 'Y Label 2']
plotter = Plotter(X, [Y_1, Y_2], Descriptions, 'X Label', YL, 'merged_plot.png')
plotter.merged_plot(Y_1max=10, Y_2max=10)

'''


'''
# Example usage:
# X = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# Y = [[2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6], [3, 4, 5, 6, 7, 8]]
# Descriptions = ['RIPE, Imbalanced', 'NYM, Imbalanced', 'NYM, Balanced']
# plotter = Plotter(X, Y, Descriptions, 'Bias parameter α', 'Entropy (bits)', 'high_quality_rectangular_plot.png')
# plotter.simple_plot(y_max=10, x_tight=True)


# Example usage:
X = [1, 2, 3, 4, 5]
Y = [[1, 4, 9, 16, 25], [2, 5, 10, 17, 26], [3, 6, 11, 18, 27], [3, 7, 11, 18, 27]]
Descriptions = ['Data 1', 'Data 2', 'Data 3', 'Data 4']
plotter = Plotter(X, Y, Descriptions, 'X-Label', 'Y-Label', 'example_plot.png')
plotter.simple_plot()
'''