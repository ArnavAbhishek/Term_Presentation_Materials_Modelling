from grain_growth_simulator import SimulationParameters, GrainGrowthSimulator
import sys

def main():
    try:
        # Initialize with 304L stainless steel parameters
        params = SimulationParameters()
        
        # Create simulator
        simulator = GrainGrowthSimulator(params)
        
        print("Starting simulation...")
        print(f"Grid size: {params.nx}x{params.ny}")
        print(f"Temperature range: {params.T_min}K - {params.T_max}K")
        print(f"Monte Carlo steps: {params.mcs_total}")
        
        # Run simulation
        k_mc_values, growth_curves, snapshots = simulator.run_simulation()
        
        # Plot results
        Q_mc = simulator.plot_results(k_mc_values, growth_curves, snapshots)
        
        print("\nSimulation Results:")
        print(f"Input Q = {params.Q/1000:.1f} kJ/mol")
        print(f"Extracted Q_MC = {Q_mc/1000:.1f} kJ/mol")
        print(f"Relative difference: {abs(Q_mc - params.Q)/params.Q*100:.1f}%")
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()