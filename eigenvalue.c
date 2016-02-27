#include "header.h"
#include <omp.h>

void run_eigenvalue(unsigned long counter, Bank *g_fission_bank, Parameters *parameters, Geometry *geometry, Material *material, Bank *source_bank, Bank *fission_bank, Tally *tally, double *keff)
{
  int i_b; // index over batches
  int i_a = -1; // index over active batches
  int i_g; // index over generations
  unsigned long i_p; // index over particles
  double keff_gen = 1; // keff of generation
  double keff_batch; // keff of batch
  double keff_mean; // keff mean over active batches
  double keff_std; // keff standard deviation over active batches

  // Loop over batches
  for(i_b=0; i_b<parameters->n_batches; i_b++){

    keff_batch = 0;

    // Turn on tallying and increment index in active batches
    if(i_b >= parameters->n_batches - parameters->n_active){
      i_a++;
      if(parameters->tally == TRUE){
        tally->tallies_on = TRUE;
      }
    }

    // Loop over generations
    for(i_g=0; i_g<parameters->n_generations; i_g++){

      // Set RNG stream for tracking
      set_stream(STREAM_TRACK);

      #pragma omp parallel for shared(i_b, i_g, parameters, geometry, material, source_bank, tally, keff_batch) private(i_p, keff_gen, fission_bank) schedule(static)
      {
        // Loop over particles
        for (i_p = 0; i_p < parameters->n_particles; i_p++) {

          // Set seed for particle i_p by skipping ahead in the random number
          // sequence stride*(total particles simulated) numbers from the initial
          // seed. This allows for reproducibility of the particle history.
          rn_skip((i_b * parameters->n_generations + i_g) * parameters->n_particles + i_p);

          // Transport the next particle
          transport(parameters, geometry, material, source_bank, fission_bank, tally, &(source_bank->p[i_p]));
        }

        // Switch RNG stream off tracking
        set_stream(STREAM_OTHER);
        rn_skip(i_b * parameters->n_generations + i_g);

        // Calculate generation k_effective and accumulate batch k_effective
        keff_gen = (double) fission_bank->n / source_bank->n;
        #pragma omp critical
        keff_batch += keff_gen;
      }

      #ifdef _OPNEMP
        //initialize first pos for memcpy
        counter = 0;

        // Sample new source particles from the particles that were added to the
        // fission bank during this generation
        synchronize_bank(counter, g_fission_bank, source_bank, fission_bank, parameters);
      #endif
    }


    // Calculate k effective
    keff_batch /= parameters->n_generations;
    if(i_a >= 0){
      keff[i_a] = keff_batch;
    }
    calculate_keff(keff, &keff_mean, &keff_std, i_a+1);

    // Tallies for this realization
    if(tally->tallies_on == TRUE){
      if(parameters->write_tally == TRUE){
        write_tally(tally, parameters->tally_file);
      }
      reset_tally(tally);
    }

    // Status text
    print_status(i_a, i_b, keff_batch, keff_mean, keff_std);
  }

  // Write out keff
  if(parameters->write_keff == TRUE){
    write_keff(keff, parameters->n_active, parameters->keff_file);
  }

  return;
}

void synchronize_bank(unsigned long counter, Bank *g_fission_bank, Bank *source_bank, Bank *fission_bank, Parameters *parameters)
{
  int n; //index over threads
  #pragma omp parallel for shared(g_fission_bank, counter, n) private(fission_bank) schedule(static)
  {
    #pragma omp for ordered
    for(n = 0; n < parameters->n_threads; n++){
      #pragma omp ordered
      {
        memcpy(&(g_fission_bank->p[counter]), fission_bank->p, fission_bank->n * sizeof(Particle));
        counter += fission_bank->n;
      }
    }
  }
  g_fission_bank->n = counter;

  unsigned long i, j;
  unsigned long n_s = source_bank->n;
  unsigned long n_f = g_fission_bank->n;

  // If the fission bank is larger than the source bank, randomly select
  // n_particles sites from the fission bank to create the new source bank
  if(n_f >= n_s){

    // Copy first n_particles sites from fission bank to source bank
    memcpy(source_bank->p, g_fission_bank->p, n_s*sizeof(Particle));

    // Replace elements with decreasing probability, such that after final
    // iteration each particle in fission bank will have equal probability of
    // being selected for source bank
    for(i=n_s; i<n_f; i++){
      j = rni(0, i+1);
      if(j<n_s){
        memcpy(&(source_bank->p[j]), &(g_fission_bank->p[i]), sizeof(Particle));
      }
    }
  }

  // If the fission bank is smaller than the source bank, use all fission bank
  // sites for the source bank and randomly sample remaining particles from
  // fission bank
  else{

    // First randomly sample particles from fission bank
    for(i=0; i<(n_s-n_f); i++){
      j = rni(0, n_f);
      memcpy(&(source_bank->p[i]), &(g_fission_bank->p[j]), sizeof(Particle));
    }

    // Fill remaining source bank sites with fission bank
    memcpy(&(source_bank->p[n_s-n_f]), g_fission_bank->p, n_f*sizeof(Particle));
  }

  g_fission_bank->n = 0;
  #pragma omp parallel for private(fission_bank)
  {
    fission_bank->n = 0;
  }

  return;
}

void calculate_keff(double *keff, double *mean, double *std, int n)
{
  int i;

  *mean = 0;
  *std = 0;

  // Calculate mean
  for(i=0; i<n; i++){
    *mean += keff[i];
  }
  *mean /= n;

  // Calculate standard deviation
  for(i=0; i<n; i++){
    *std += pow(keff[i] - *mean, 2);
  }
  *std = sqrt(*std/(n-1));

  return;
}
