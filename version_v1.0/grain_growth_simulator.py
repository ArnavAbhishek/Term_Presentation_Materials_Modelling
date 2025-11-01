import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from skimage import measure
from dataclasses import dataclass
from typing import Tuple, List, Dict
import seaborn as sns

@dataclass
class SimulationParameters:
    """Physical and simulation parameters"""
    # Physical constants
    kB: float = 1.380649e-23  # Boltzmann constant (J/K)
    R: float = 8.314462618    # Gas constant (J/mol·K)
    
    # Material parameters (304L stainless steel)
    Q: float = 118000.0       # Activation energy (J/mol)
    Tm: float = 1723.0        # Melting temperature (K)
    gamma_GB: float = 1.4     # Grain boundary energy (J/m²)
    Z_GB: float = 8.96e19     # Atomic density at GB (atoms/m²)
    
    # Simulation parameters
    nx: int = 150            # Increased for better statistics
    ny: int = 150
    n_orientations: int = 20  # Fewer orientations for clearer visualization
    k_length: float = 2.5e-6
    
    # Monte Carlo parameters
    n_temps: int = 10
    T_min: float = 1000.0    # Higher temperature for faster growth
    T_max: float = 1900.0
    mcs_total: int = 2000    # More steps for better growth
    sample_interval: int = 100

class GrainGrowthSimulator:
    def __init__(self, params: SimulationParameters):
        self.p = params
        self.temperatures = np.linspace(self.p.T_min, self.p.T_max, self.p.n_temps)
        self.lattice = None
        self.initialize_lattice()
        
    def initialize_lattice(self):
        """Initialize random grain orientations"""
        self.lattice = np.random.randint(1, self.p.n_orientations + 1, 
                                       size=(self.p.nx, self.p.ny), 
                                       dtype=np.int32)
    
    def get_neighbors(self, i: int, j: int, neighborhood: int = 4) -> List[Tuple[int, int]]:
        """Get valid neighbor indices"""
        neighbors = []
        if neighborhood == 4:
            candidates = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
        else:  # neighborhood == 8
            candidates = [(i-1,j), (i+1,j), (i,j-1), (i,j+1),
                        (i-1,j-1), (i-1,j+1), (i+1,j-1), (i+1,j+1)]
            
        for ii, jj in candidates:
            if 0 <= ii < self.p.nx and 0 <= jj < self.p.ny:
                neighbors.append((ii, jj))
        return neighbors
    
    def calculate_energy_change(self, site: Tuple[int, int], new_orientation: int) -> float:
        """Calculate energy change with correct neighbor counting"""
        i, j = site
        old_orientation = self.lattice[i,j]
        neighbors = self.get_neighbors(i, j, neighborhood=8)  # Use 8 neighbors
        
        E_before = E_after = 0.0
        for ii, jj in neighbors:
            neighbor_orientation = self.lattice[ii,jj]
            # Only count different orientations
            if old_orientation != neighbor_orientation:
                E_before += self.p.gamma_GB * self.p.k_length
            if new_orientation != neighbor_orientation:
                E_after += self.p.gamma_GB * self.p.k_length
        
        return E_after - E_before
    
    def monte_carlo_step(self, temperature: float) -> None:
        """Perform one Monte Carlo step with corrected energy scaling"""
        # Calculate mobility term
        P_mob = np.exp(-self.p.Q / self.p.R * (1/temperature - 1/self.p.Tm))
        
        # Add energy scaling factor
        kT = self.p.kB * temperature
        
        for _ in range(self.p.nx * self.p.ny):
            i = np.random.randint(0, self.p.nx)
            j = np.random.randint(0, self.p.ny)
            
            neighbors = self.get_neighbors(i, j, neighborhood=8)  # Use 8 neighbors
            if not neighbors:
                continue
            
            ii, jj = neighbors[np.random.randint(len(neighbors))]
            new_orientation = self.lattice[ii,jj]
            
            if new_orientation == self.lattice[i,j]:
                continue
            
            dE = self.calculate_energy_change((i,j), new_orientation)
            
            # Simplified acceptance criterion with correct energy scaling
            if dE <= 0:
                self.lattice[i,j] = new_orientation
            else:
                P = P_mob * np.exp(-dE / kT)  # Remove Z_GB scaling
                if np.random.random() < P:
                    self.lattice[i,j] = new_orientation
    
    def analyze_grain_statistics(self) -> Dict:
        """Analyze current grain structure"""
        labeled = measure.label(self.lattice, connectivity=2)
        regions = measure.regionprops(labeled)
        
        areas = [r.area * (self.p.k_length**2) for r in regions]
        mean_size = np.mean(areas)
        std_size = np.std(areas)
        n_grains = len(regions)
        
        return {
            'mean_size': mean_size,
            'std_size': std_size,
            'n_grains': n_grains,
            'areas': areas
        }
    
    def run_simulation(self) -> Tuple[np.ndarray, Dict, Dict]:
        """Run full simulation across temperature range"""
        k_mc_values = []
        growth_curves = {}
        snapshots = {}
        
        for T in self.temperatures:
            print(f"Simulating T = {T:.1f}K")
            self.initialize_lattice()
            snapshots[T] = []
            
            sizes = []
            times = []
            
            # Store initial, middle and final states
            snapshot_points = [0, self.p.mcs_total//2, self.p.mcs_total]
            
            for step in range(self.p.mcs_total + 1):
                if step % self.p.sample_interval == 0:
                    stats = self.analyze_grain_statistics()
                    sizes.append(stats['mean_size'])
                    times.append(step)
                
                if step in snapshot_points:
                    snapshots[T].append((step, self.lattice.copy()))
                
                self.monte_carlo_step(T)
            
            # Calculate growth rate
            slope, _ = np.polyfit(times, sizes, 1)
            k_mc_values.append(slope)
            growth_curves[T] = (np.array(times), np.array(sizes))
        
        return np.array(k_mc_values), growth_curves, snapshots

    def plot_results(self, k_mc_values: np.ndarray, growth_curves: Dict, snapshots: Dict) -> float:
        """Plot simulation results with improved visualization"""
        # Create figure for microstructure evolution
        plt.figure(figsize=(15, 5))
        
        # Plot microstructure evolution at middle temperature
        mid_temp = self.temperatures[len(self.temperatures)//2]
        snapshots_mid_T = snapshots[mid_temp]
        
        for idx, (step, lattice) in enumerate(snapshots_mid_T):
            plt.subplot(1, 3, idx+1)
            labeled = measure.label(lattice, connectivity=2)
            
            # Use a better colormap for grain visualization
            plt.imshow(labeled, cmap='nipy_spectral')
            plt.title(f'T={mid_temp:.0f}K\nMCS={step}\nGrains={np.max(labeled)}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Create figure for growth curves and Arrhenius plot
        plt.figure(figsize=(12, 5))
        
        # Growth curves
        plt.subplot(121)
        for T, (times, sizes) in growth_curves.items():
            plt.plot(times, sizes, 'o-', label=f'{T:.0f}K')
        plt.xlabel('Time (MCS)')
        plt.ylabel('Mean Grain Size (m²)')
        plt.title('Grain Growth Kinetics')
        plt.legend()
        
        # Arrhenius plot
        plt.subplot(122)
        inv_T = 1000/self.temperatures
        ln_k = np.log(k_mc_values)
        slope, intercept = np.polyfit(inv_T, ln_k, 1)
        Q_mc = -slope * self.p.R
        
        plt.plot(inv_T, ln_k, 'bo-', label='Data')
        plt.plot(inv_T, slope*inv_T + intercept, 'r--', 
                label=f'Q_MC = {Q_mc/1000:.1f} kJ/mol')
        plt.xlabel('1000/T (K⁻¹)')
        plt.ylabel('ln(K)')
        plt.title('Arrhenius Plot')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return Q_mc