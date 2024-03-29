// CDDL HEADER START
//
// The contents of this file are subject to the terms of the Common Development
// and Distribution License Version 1.0 (the "License").
//
// You can obtain a copy of the license at
// http://www.opensource.org/licenses/CDDL-1.0.  See the License for the
// specific language governing permissions and limitations under the License.
//
// When distributing Covered Code, include this CDDL HEADER in each file and
// include the License file in a prominent location with the name LICENSE.CDDL.
// If applicable, add the following below this CDDL HEADER, with the fields
// enclosed by brackets "[]" replaced with your own identifying information:
//
// Portions Copyright (c) [yyyy] [name of copyright owner]. All rights reserved.
//
// CDDL HEADER END
//

//
// Copyright (c) 2021, Regents of the University of Minnesota.
// All rights reserved.
//
// Contributors:
//    Daniel S. Karls
//
// Based on the LennardJones_Ar example portable model distributed with the KIM
// API, authored by Ryan S. Elliott:
//
//  https://github.com/openkim/kim-api/tree/cd2fe213feb5dacfb84167c1b328f37b5914f739/examples/portable-models/LennardJones_Ar
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "ml_model.hpp"

#include "KIM_LogMacros.hpp"
#include "KIM_ModelHeaders.hpp"

#define DIMENSION 3

// Parameters used for performing initial allocation of the array containing
// the number of neighbors of each atom and the array used to contain the
// neighbor lists of every atom.  These will be dynamic arrays that are grown
// as necessary as new configurations are encountered throughout simulation.
#define NUM_ATOMS_FOR_INITIAL_ARRAY_ALLOCATION 1000
#define NUM_NEIGHBORS_PER_ATOM_FOR_INITIAL_ARRAY_ALLOCATION 50

#define KIM_DEVICE_ENV_VAR "KIM_MODEL_EXECUTION_DEVICE"

namespace
{
    class KIMMLModel
    {
    public:
        //****************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelCreate
        KIMMLModel(KIM::ModelCreate *const modelCreate,
                   KIM::LengthUnit const requestedLengthUnit,
                   KIM::EnergyUnit const requestedEnergyUnit,
                   KIM::ChargeUnit const requestedChargeUnit,
                   KIM::TemperatureUnit const requestedTemperatureUnit,
                   KIM::TimeUnit const requestedTimeUnit, int *const error)
            : modelWillNotRequestNeighborsOfNoncontributingParticles_(1)
        {
            *error = ConvertUnits(modelCreate, requestedLengthUnit,
                                  requestedEnergyUnit, requestedChargeUnit,
                                  requestedTemperatureUnit, requestedTimeUnit);
            if (*error)
                return;

            modelCreate->SetModelNumbering(KIM::NUMBERING::zeroBased);

            // FIXME: Define influence distance and cutoff -- should be defined
            // by each model in a parameter file once this is a driver
            influenceDistance_ = 3.77118;
            modelCreate->SetInfluenceDistancePointer(&influenceDistance_);

            cutoff_ = influenceDistance_;
            modelCreate->SetNeighborListPointers(
                1, &cutoff_,
                &modelWillNotRequestNeighborsOfNoncontributingParticles_);

            // FIXME: When we make this into a model driver, read the species
            // from a parameter file stored with the model
            modelCreate->SetSpeciesCode(KIM::SPECIES_NAME::Si, 0);

            // use function pointer declarations to verify prototypes
            KIM::ModelComputeArgumentsCreateFunction *CACreate =
                KIMMLModel::ComputeArgumentsCreate;
            KIM::ModelComputeFunction *compute = KIMMLModel::Compute;
            KIM::ModelComputeArgumentsDestroyFunction *CADestroy =
                KIMMLModel::ComputeArgumentsDestroy;
            KIM::ModelDestroyFunction *destroy = KIMMLModel::Destroy;

            *error =
                modelCreate->SetRoutinePointer(
                    KIM::MODEL_ROUTINE_NAME::ComputeArgumentsCreate,
                    KIM::LANGUAGE_NAME::cpp, true,
                    reinterpret_cast<KIM::Function *>(CACreate)) ||
                modelCreate->SetRoutinePointer(
                    KIM::MODEL_ROUTINE_NAME::Compute, KIM::LANGUAGE_NAME::cpp,
                    true, reinterpret_cast<KIM::Function *>(compute)) ||
                modelCreate->SetRoutinePointer(
                    KIM::MODEL_ROUTINE_NAME::ComputeArgumentsDestroy,
                    KIM::LANGUAGE_NAME::cpp, true,
                    reinterpret_cast<KIM::Function *>(CADestroy)) ||
                modelCreate->SetRoutinePointer(
                    KIM::MODEL_ROUTINE_NAME::Destroy, KIM::LANGUAGE_NAME::cpp,
                    true, reinterpret_cast<KIM::Function *>(destroy));
            if (*error)
                return;

            // FIXME: Once this is a model driver, read the parameter file from
            // the parameterized model
            const char *model = "SW_energy_and_forces.pt";

            // Create ML wrapper object
            LOG_DEBUG("Creating ML framework wrapper object");

            ml_model_ = MLModel::create(model, ML_MODEL_PYTORCH,
                                        std::getenv(KIM_DEVICE_ENV_VAR));
            LOG_DEBUG("Done creating ML framework wrapper object");

            // Create initial empty arrays for num_neighbors and neighbor_list
            // arrays on heap.  These are retained in memory and are enlarged as
            // necessary to be able to fit the necessary data for any atomic
            // configuration for which compute() is called
            num_neighbors_.reserve(NUM_ATOMS_FOR_INITIAL_ARRAY_ALLOCATION);

            // Raveled neighbor list for each atom in the entire current
            // configuration
            neighbor_list_.reserve(
                NUM_ATOMS_FOR_INITIAL_ARRAY_ALLOCATION *
                NUM_NEIGHBORS_PER_ATOM_FOR_INITIAL_ARRAY_ALLOCATION);

            // Indicate no errors occurred
            *error = false;

            return;
        };

        //****************************************************************************
        ~KIMMLModel()
        {
            if (ml_model_ != nullptr)
            {
                delete ml_model_;
            }
        };

//****************************************************************************
// no need to make these "extern" since KIM will only access them
// via function pointers.  "static" is required so that there is not
// an implicit this pointer added to the prototype by the C++ compiler
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDestroy
        static int Destroy(KIM::ModelDestroy *const modelDestroy)
        {
            KIMMLModel *model_buffer;
            modelDestroy->GetModelBufferPointer(
                reinterpret_cast<void **>(&model_buffer));

            LOG_DEBUG("Deallocating model buffer");
            if (model_buffer != nullptr)
            {
                // FIXME: Will deallocating the MLModel object prevent other
                // active model objects from continuing their inference?  I
                // guess this is just the ML wrapper object, so it will just
                // end up decrementing the ML object's reference counter?
                delete model_buffer;
            }

            // Return false to indicate no errors occurred
            return false;
        }

//****************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelCompute
        static int
        Compute(KIM::ModelCompute const *const modelCompute,
                KIM::ModelComputeArguments const *const modelComputeArguments)
        {
            int *numberOfParticlesPointer;
            int *particleSpeciesCodes; // FIXME: Implement species code handling
            int *particleContributing;
            double *coordinates;
            double *partialEnergy;
            double *partialForces;
            KIMMLModel *model_buffer;

            modelCompute->GetModelBufferPointer(
                reinterpret_cast<void **>(&model_buffer));

            int error = modelComputeArguments->GetArgumentPointer(
                            KIM::COMPUTE_ARGUMENT_NAME::numberOfParticles,
                            &numberOfParticlesPointer) ||
                        modelComputeArguments->GetArgumentPointer(
                            KIM::COMPUTE_ARGUMENT_NAME::particleSpeciesCodes,
                            &particleSpeciesCodes) ||
                        modelComputeArguments->GetArgumentPointer(
                            KIM::COMPUTE_ARGUMENT_NAME::particleContributing,
                            &particleContributing) ||
                        modelComputeArguments->GetArgumentPointer(
                            KIM::COMPUTE_ARGUMENT_NAME::coordinates,
                            (double const **)&coordinates) ||
                        modelComputeArguments->GetArgumentPointer(
                            KIM::COMPUTE_ARGUMENT_NAME::partialEnergy,
                            &partialEnergy) ||
                        modelComputeArguments->GetArgumentPointer(
                            KIM::COMPUTE_ARGUMENT_NAME::partialForces,
                            (double const **)&partialForces);

            if (error)
            {
                LOG_ERROR("Unable to get argument pointers");
                return error;
            }

            int const numberOfParticles = *numberOfParticlesPointer;

            // IMPORTANT: We require a specific ordering of the arguments to
            // the pytorch model's `forward` method:
            //
            //   forward(self, particle_contributing, coordinates,
            //                 number_of_neighbors, neighbor_list)
            //
            model_buffer->ml_model_->SetInputNode(0, particleContributing,
                                                  numberOfParticles);
            model_buffer->ml_model_->SetInputNode(1, coordinates,
                                                  3 * numberOfParticles, true);

            // TODO: We could save a decent bit of work if there were a way for
            // the model to know whether the neighbor list has changed since
            // the last call to its Compute.  Of course, the model could do
            // this itself by checking the neighbor skin, but perhaps a
            // mechanism in the KIM API is possible.
            model_buffer->update_neighbor_arrays(numberOfParticles,
                                                 modelComputeArguments);

            model_buffer->ml_model_->SetInputNode(
                2, model_buffer->num_neighbors_.data(),
                model_buffer->num_neighbors_.size());
            model_buffer->ml_model_->SetInputNode(
                3, model_buffer->neighbor_list_.data(),
                model_buffer->neighbor_list_.size());

            // TODO: Generalize the Run method so that we can get more than a
            // hard-coded set of outputs back?

            // IMPORTANT: We also require that the pytorch model's `forward`
            // method return a tuple where the energy is the first entry and
            // the forces are the second
            model_buffer->ml_model_->Run(partialEnergy, partialForces);

            // Return false to indicate no errors occurred
            return false;
        };

        //****************************************************************************
        static int ComputeArgumentsCreate(
            KIM::ModelCompute const *const /* modelCompute */,
            KIM::ModelComputeArgumentsCreate *const modelComputeArgumentsCreate)
        {
            // register arguments
            int error = modelComputeArgumentsCreate->SetArgumentSupportStatus(
                            KIM::COMPUTE_ARGUMENT_NAME::partialEnergy,
                            KIM::SUPPORT_STATUS::required) ||
                        modelComputeArgumentsCreate->SetArgumentSupportStatus(
                            KIM::COMPUTE_ARGUMENT_NAME::partialForces,
                            KIM::SUPPORT_STATUS::required);

            // register callbacks
            //
            // none

            return error;
        }

        //****************************************************************************
        static int ComputeArgumentsDestroy(
            KIM::ModelCompute const *const /* modelCompute */,
            KIM::ModelComputeArgumentsDestroy *const
            /* modelComputeArgumentsDestroy */)
        {
            // nothing further to do

            return false;
        }

    private:
        MLModel *ml_model_;

        // The number of neighbors for each atom in the entire current
        // configuration
        std::vector<int> num_neighbors_;

        // Raveled neighbor list for each atom in the entire current
        // configuration
        std::vector<int> neighbor_list_;

        double influenceDistance_;
        double cutoff_;
        int const modelWillNotRequestNeighborsOfNoncontributingParticles_;

        void update_neighbor_arrays(
            int number_of_particles,
            KIM::ModelComputeArguments const *const modelComputeArguments)
        {
            int number_of_neighbors;
            int const *neighbors;

            // Clear num_neighbors and neighbor_list arrays in model buffer
            // and populate.
            num_neighbors_.clear();
            neighbor_list_.clear();

            for (int atom_i = 0; atom_i < number_of_particles; ++atom_i)
            {
                // Get number of neighbors for this atom and concatenate it
                // onto the global number-of-neighbors list.  Then,
                // concatenate its actual neighbors onto the global neighbor
                // list
                modelComputeArguments->GetNeighborList(
                    0, atom_i, &number_of_neighbors, &neighbors);

                // FIXME: Automatic vector resizing is probably slow
                // compared to doing a doubling of the underlying array size
                // ourselves!
                num_neighbors_.push_back(number_of_neighbors);

                for (int neigh = 0; neigh < number_of_neighbors; ++neigh)
                {
                    // FIXME: Automatic vector resizing is probably slow
                    // compared to doing a doubling of the underlying array
                    // size ourselves!
                    neighbor_list_.push_back(neighbors[neigh]);
                }
            }
        }

        //****************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelCreate
        int ConvertUnits(KIM::ModelCreate *const modelCreate,
                         KIM::LengthUnit const requestedLengthUnit,
                         KIM::EnergyUnit const requestedEnergyUnit,
                         KIM::ChargeUnit const requestedChargeUnit,
                         KIM::TemperatureUnit const requestedTemperatureUnit,
                         KIM::TimeUnit const requestedTimeUnit)
        {
            int ier;

            // define default base units
            KIM::LengthUnit fromLength = KIM::LENGTH_UNIT::A;
            KIM::EnergyUnit fromEnergy = KIM::ENERGY_UNIT::eV;
            KIM::ChargeUnit fromCharge = KIM::CHARGE_UNIT::unused;
            KIM::TemperatureUnit fromTemperature =
                KIM::TEMPERATURE_UNIT::unused;
            KIM::TimeUnit fromTime = KIM::TIME_UNIT::unused;

            // changing units of cutoffs and sigmas
            double convertLength = 1.0;
            ier = KIM::ModelCreate::ConvertUnit(
                fromLength, fromEnergy, fromCharge, fromTemperature, fromTime,
                requestedLengthUnit, requestedEnergyUnit, requestedChargeUnit,
                requestedTemperatureUnit, requestedTimeUnit, 1.0, 0.0, 0.0, 0.0,
                0.0, &convertLength);
            if (ier)
            {
                LOG_ERROR("Unable to convert length unit");
                return ier;
            }
            influenceDistance_ *= convertLength; // convert to active units
            cutoff_ = influenceDistance_;

            // FIXME: Do unit conversion

            // register units
            ier = modelCreate->SetUnits(
                requestedLengthUnit, requestedEnergyUnit,
                KIM::CHARGE_UNIT::unused, KIM::TEMPERATURE_UNIT::unused,
                KIM::TIME_UNIT::unused);
            if (ier)
            {
                LOG_ERROR("Unable to set units to requested values");
                return ier;
            }

            // everything is good
            ier = false;
            return ier;
        }
    }; // End class definition
} // End anonymous namespace

extern "C"
{
    //******************************************************************************
    int model_create(KIM::ModelCreate *const modelCreate,
                     KIM::LengthUnit const requestedLengthUnit,
                     KIM::EnergyUnit const requestedEnergyUnit,
                     KIM::ChargeUnit const requestedChargeUnit,
                     KIM::TemperatureUnit const requestedTemperatureUnit,
                     KIM::TimeUnit const requestedTimeUnit)
    {
        int error;

        // Create model class object.  This will allocate a ML wrapper object
        // and store it in an instance attr on the model class object.
        KIMMLModel *const model_buffer =
            new KIMMLModel(modelCreate, requestedLengthUnit,
                           requestedEnergyUnit, requestedChargeUnit,
                           requestedTemperatureUnit, requestedTimeUnit, &error);

        modelCreate->SetModelBufferPointer(static_cast<void *>(model_buffer));

        if (error)
        {
            // constructor already reported the error
            delete model_buffer;
            return error;
        }

        // everything is good
        return false;
    }
} // extern "C"
