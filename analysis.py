import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import os

class CraterAnalyzer:
    def __init__(self, meters_per_pixel=1.0):
        """Initialize crater analyzer with scale"""
        self.meters_per_pixel = meters_per_pixel
        
    def analyze_craters(self, sam_output, save_dir=None):
        """
        Perform scientific analysis of crater data
        
        Args:
            sam_output: Dictionary from SAM containing masks and crater data
            save_dir: Directory to save analysis plots
            
        Returns:
            pandas.DataFrame: Processed crater data with real-world measurements
        """
        masks = sam_output['masks']
        crater_data = sam_output['crater_data']
        image_rgb = sam_output['image_rgb']
        
        if len(crater_data) == 0:
            print("No crater data to analyze")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(crater_data)
        
        # Add real-world measurements
        df["Diameter_m"] = df["Diameter_px"] * self.meters_per_pixel
        df["Area_m2"] = df["Area_px"] * (self.meters_per_pixel**2)
        
        print(f"Scale applied: 1 pixel = {self.meters_per_pixel} meters")
        print(f"Analyzed {len(df)} craters")
        
        # Generate analysis plots
        if save_dir:
            self._create_analysis_plots(df, masks, image_rgb, save_dir)
        
        return df
    
    def _create_analysis_plots(self, df, masks, image_rgb, save_dir):
        """Create and save scientific analysis plots"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Histogram of crater diameters
        plt.figure(figsize=(10, 6))
        plt.hist(df["Diameter_m"], bins=min(10, len(df)), color="skyblue", edgecolor="black")
        plt.title("Crater Diameter Distribution", fontsize=14)
        plt.xlabel("Diameter (meters)", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "diameter_distribution.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Scatter plot: Area vs Diameter
        plt.figure(figsize=(10, 6))
        plt.scatter(df["Diameter_m"], df["Area_m2"], c="purple", alpha=0.7, s=60)
        plt.title("Crater Area vs Diameter", fontsize=14)
        plt.xlabel("Diameter (m)", fontsize=12)
        plt.ylabel("Area (m²)", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df["Diameter_m"], df["Area_m2"], 1)
        p = np.poly1d(z)
        plt.plot(df["Diameter_m"], p(df["Diameter_m"]), "r--", alpha=0.8, label="Trend line")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "area_vs_diameter.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Spatial distribution
        plt.figure(figsize=(12, 10))
        plt.imshow(image_rgb)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(masks)))
        
        for i, mask in enumerate(masks):
            coords = np.column_stack(np.where(mask))
            y_mean, x_mean = coords.mean(axis=0)
            plt.scatter(x_mean, y_mean, marker="o", c=[colors[i]], s=100, 
                       edgecolors='black', linewidth=2, label=f"Crater {i+1}")
        
        plt.title("Spatial Distribution of Craters", fontsize=14)
        plt.axis("off")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "spatial_distribution.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Density heatmap
        if len(masks) > 0:
            density_map = np.zeros(masks[0].shape)
            for mask in masks:
                density_map += mask.astype(int)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(image_rgb, alpha=0.6)
            im = plt.imshow(density_map, cmap="hot", alpha=0.7)
            plt.colorbar(im, label="Crater Density", shrink=0.8)
            plt.title("Crater Density Heatmap", fontsize=14)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "density_heatmap.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 5. Statistical summary plot
        plt.figure(figsize=(12, 8))
        
        # Create subplots for statistics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Diameter statistics
        ax1.boxplot(df["Diameter_m"])
        ax1.set_title("Diameter Distribution")
        ax1.set_ylabel("Diameter (m)")
        
        # Area statistics
        ax2.boxplot(df["Area_m2"])
        ax2.set_title("Area Distribution")
        ax2.set_ylabel("Area (m²)")
        
        # Size classification
        small = df[df["Diameter_m"] < df["Diameter_m"].mean()]
        large = df[df["Diameter_m"] >= df["Diameter_m"].mean()]
        
        sizes = [len(small), len(large)]
        labels = [f'Small (<{df["Diameter_m"].mean():.1f}m)', f'Large (≥{df["Diameter_m"].mean():.1f}m)']
        ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title("Crater Size Classification")
        
        # Summary statistics text
        stats_text = f"""
        Total Craters: {len(df)}
        
        Diameter (m):
        Mean: {df['Diameter_m'].mean():.2f}
        Std: {df['Diameter_m'].std():.2f}
        Min: {df['Diameter_m'].min():.2f}
        Max: {df['Diameter_m'].max():.2f}
        
        Area (m²):
        Mean: {df['Area_m2'].mean():.2f}
        Total: {df['Area_m2'].sum():.2f}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title("Summary Statistics")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "statistical_summary.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Analysis plots saved to: {save_dir}")

def main():
    # Test analyzer independently
    print("Crater analyzer module loaded successfully!")

if __name__ == "__main__":
    main()