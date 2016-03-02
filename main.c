#include "header.h"
#include <omp.h>

#ifdef _OPENMP
Bank *fission_bank;
#endif

int main(int argc, char *argv[])
{
  Parameters *parameters; // user defined parameters
  Geometry *geometry; // homogenous cube geometry
  Material *material; // problem material
  Bank *source_bank; // array for particle source sites
  Tally *tally; // scalar flux tally
  double *keff; // effective multiplication factor
  double t1, t2; // timers

  #ifdef _OPENMP
    unsigned long counter = 0; //counter to decide the start pos of master bank copy from sub banks
    Bank *g_fission_bank; //global fission bank
  #endif

  // Get inputs: set parameters to default values, parse parameter file,
  // override with any command line inputs, and print parameters
  parameters = init_parameters();
  parse_parameters(parameters);
  read_CLI(argc, argv, parameters);
  print_parameters(parameters);

  // Set initial RNG seed
  set_initial_seed(parameters->seed);
  set_stream(STREAM_INIT);

  // Create files for writing results to
  init_output(parameters);

  // Set up geometry
  geometry = init_geometry(parameters);

  // Set up material
  material = init_material(parameters);

  // Set up tallies
  tally = init_tally(parameters);

  // Create source bank and initial source distribution
  source_bank = init_source_bank(parameters, geometry);

  // Create fission bank
  #ifdef _OPENMP
    omp_set_num_threads(parameters->n_threads); // Set number of openmp threads

    // Allocate one master fission bank
    g_fission_bank = init_bank(2*parameters->n_particles);
  #endif

  // Set up array for k effective
  keff = calloc(parameters->n_active, sizeof(double));

  center_print("SIMULATION", 79);
  border_print();
  printf("%-15s %-15s %-15s\n", "BATCH", "KEFF", "MEAN KEFF");

  #ifdef _OPENMP
    // Start time
    t1 = omp_get_wtime();

    run_eigenvalue(counter, g_fission_bank, parameters, geometry, material, source_bank, fission_bank, tally, keff);

    // Stop time
    t2 = omp_get_wtime();
  #endif

  printf("Simulation time: %f secs\n", t2-t1);

  // Free memory
  #ifdef _OPENMP
    free_bank(g_fission_bank);
  #endif

  free(keff);
  free_tally(tally);
  free_bank(source_bank);
  free_material(material);
  free(geometry);
  free(parameters);

  return 0;
}
