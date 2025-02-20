import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import random
from joblib import Parallel, delayed  # For parallel processing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class BacteriaModel:
    """Represents a single 'bacterium' (ML model) with its hyperparameters and architecture."""

    def __init__(self, model_type='xgboost', params=None, architecture=None):
        self.model_type = model_type
        self.params = params if params else self.generate_random_params()
        self.architecture = architecture if architecture else self.generate_random_architecture()  # For NN
        self.model = None
        self.fitness = -1
        self.similarity_penalty = 0  # For diversity control

    def generate_random_params(self):
        """Generates random hyperparameters based on the model type."""
        if self.model_type == 'xgboost':
            return {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'eta': random.uniform(0.01, 0.3),
                'max_depth': random.randint(3, 10),
                'subsample': random.uniform(0.5, 1.0),
                'colsample_bytree': random.uniform(0.5, 1.0),
                'min_child_weight': random.randint(1, 10),
                'gamma': random.uniform(0, 0.5),
                'lambda': random.uniform(0, 2),
                'alpha': random.uniform(0, 1),
                'seed': random.randint(0, 1000)
            }
        elif self.model_type == 'keras':
            return {
                'learning_rate': random.uniform(0.0001, 0.01),
                'dropout_rate': random.uniform(0.0, 0.5),
                'optimizer': random.choice(['adam', 'sgd', 'rmsprop']),
                'batch_size' : random.choice([16,32,64])
            }
        else:
            raise ValueError("Unsupported model type.")

    def generate_random_architecture(self):
        """Generates a random architecture for Keras models."""
        if self.model_type == 'keras':
            num_layers = random.randint(1, 4)  # Between 1 and 4 hidden layers
            architecture = []
            for _ in range(num_layers):
                architecture.append({
                    'units': random.randint(16, 128),
                    'activation': random.choice(['relu', 'tanh', 'sigmoid'])
                })
            return architecture
        else:
            return None  # No architecture for XGBoost

    def build_model(self, input_shape):
      """Builds the Keras model based on the architecture."""
      if self.model_type == 'keras':
          model = Sequential()
          # Input layer
          if len(self.architecture) > 0: #Added a check for architectures
            model.add(Dense(self.architecture[0]['units'], activation=self.architecture[0]['activation'], input_shape=(input_shape,)))
          else: # If empty architecture, put something basic
             model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
          model.add(Dropout(self.params['dropout_rate']))

          # Hidden layers
          for layer_params in self.architecture[1:]:
              model.add(Dense(layer_params['units'], activation=layer_params['activation']))
              model.add(Dropout(self.params['dropout_rate']))

          # Output layer
          model.add(Dense(1, activation='sigmoid'))  # Binary classification

          if self.params['optimizer'] == 'adam':
              optimizer = Adam(learning_rate=self.params['learning_rate'])
          elif self.params['optimizer'] == 'sgd':
              optimizer = tf.keras.optimizers.SGD(learning_rate=self.params['learning_rate'])
          else:
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.params['learning_rate'])


          model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
          self.model = model
      else:
        raise ValueError("Build Model only applicable to keras")


    def train(self, X_train, y_train, X_val, y_val):
      """Trains the model and evaluates its fitness."""
      if self.model_type == 'xgboost':
          dtrain = xgb.DMatrix(X_train, label=y_train)
          dval = xgb.DMatrix(X_val, label=y_val)
          self.model = xgb.train(self.params, dtrain, num_boost_round=100,
                                  evals=[(dval, 'validation')], early_stopping_rounds=10, verbose_eval=False)
          preds = self.model.predict(dval)
          best_preds = np.asarray([np.round(value) for value in preds])
          self.fitness = accuracy_score(y_val, best_preds)

      elif self.model_type == 'keras':
          if self.model is None: #Added a check to ensure model is built
            self.build_model(X_train.shape[1])
          self.model.fit(X_train, y_train, epochs=50, batch_size=self.params['batch_size'],
                          validation_data=(X_val, y_val), verbose=0,
                          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]) #Early Stopping

          loss, accuracy = self.model.evaluate(X_val, y_val, verbose=0)
          self.fitness = accuracy

      else:
        raise ValueError("Invalid Model Type")

    def predict(self, X):
        """Makes predictions using the trained model."""
        if self.model_type == 'xgboost':
            dtest = xgb.DMatrix(X)
            return self.model.predict(dtest)
        elif self.model_type == 'keras':
            return self.model.predict(X)
        else:
            raise ValueError("Unsupported model type.")

    def get_params(self):
        """Returns the hyperparameters."""
        return self.params

    def get_architecture(self):
        """Returns the architecture (for Keras models)."""
        return self.architecture

    def set_params(self, params):
        """Sets the hyperparameters."""
        self.params = params

    def set_architecture(self, architecture):
        """Sets the architecture (for Keras models)."""
        self.architecture = architecture
        self.model = None # Reset model so it can be rebuilt

    def mutate(self, mutation_prob=0.1):
        """Introduces random mutations in the hyperparameters and architecture."""

        # Mutate hyperparameters
        for key, value in self.params.items():
            if random.random() < mutation_prob:
                if isinstance(value, float):
                    # Apply a small random change (e.g., +/- 10%)
                    self.params[key] = value * (1 + random.uniform(-0.1, 0.1))
                elif isinstance(value, int):
                    # Change to a nearby integer value
                    self.params[key] = value + random.choice([-1, 1])
                elif isinstance(value, str):
                    if key == 'optimizer':
                        self.params[key] = random.choice(['adam', 'sgd', 'rmsprop'])


        # Mutate architecture (for Keras models)
        if self.model_type == 'keras' and self.architecture:
            if random.random() < mutation_prob:
                # Add or remove a layer
                if random.random() < 0.5 and len(self.architecture) > 1:  # Remove a layer (but not the only layer)
                    self.architecture.pop(random.randrange(len(self.architecture)))
                elif len(self.architecture) < 5: # Limit Max Layers
                    # Add a layer
                    new_layer = {
                        'units': random.randint(16, 128),
                        'activation': random.choice(['relu', 'tanh', 'sigmoid'])
                    }
                    self.architecture.insert(random.randrange(len(self.architecture) + 1), new_layer)
            # Mutate Existing layer
            if random.random() < mutation_prob:
              layer_to_mutate = random.choice(self.architecture)
              if 'units' in layer_to_mutate:
                layer_to_mutate['units'] = random.randint(16,128)
              if 'activation' in layer_to_mutate:
                layer_to_mutate['activation'] = random.choice(['relu', 'tanh', 'sigmoid'])
            self.model = None # Reset so new architecture is built


def conjugation(bacteria1, bacteria2, exchange_prob=0.5):
    """Implements conjugation (hyperparameter and architecture exchange)."""
    new_params1 = bacteria1.get_params().copy()
    new_params2 = bacteria2.get_params().copy()
    new_arch1 = (bacteria1.get_architecture().copy() if bacteria1.get_architecture() else None) # Added architecture copy
    new_arch2 = (bacteria2.get_architecture().copy() if bacteria2.get_architecture() else None)


    for key in new_params1:
        if random.random() < exchange_prob:
            new_params1[key], new_params2[key] = new_params2[key], new_params1[key]

    # Exchange architecture components (if Keras models)
    if bacteria1.model_type == 'keras' and bacteria2.model_type == 'keras':
        if random.random() < exchange_prob:  # Exchange entire architectures with a probability
            new_arch1, new_arch2 = new_arch2, new_arch1
        else: #Exchange specific layers
          if random.random() < exchange_prob and new_arch1 and new_arch2: # Added condition for non-empty
            index1 = random.randrange(len(new_arch1))
            index2 = random.randrange(len(new_arch2))
            new_arch1[index1], new_arch2[index2] = new_arch2[index2], new_arch1[index1] # Swap the layers

    b1 = BacteriaModel(bacteria1.model_type, new_params1, new_arch1)
    b2 = BacteriaModel(bacteria2.model_type, new_params2, new_arch2)

    return b1,b2



def transformation(bacteria, graveyard, uptake_prob=0.2):
    """Implements transformation (parameter and architecture borrowing)."""
    if not graveyard:
        return bacteria

    dead_bacteria = random.choice(graveyard)
    new_params = bacteria.get_params().copy()
    new_arch = (bacteria.get_architecture().copy() if bacteria.get_architecture() else None) # Added architecture copy

    for key in new_params:
        if random.random() < uptake_prob:
            new_params[key] = dead_bacteria.get_params()[key]

    # Borrow architecture components (if Keras models and dead bacteria was also Keras)
    if bacteria.model_type == 'keras' and dead_bacteria.model_type == 'keras':
        if random.random() < uptake_prob:
            if dead_bacteria.get_architecture() and new_arch: # Check for non-empty
              # Copy a random layer from the dead bacteria
              layer_to_copy = random.choice(dead_bacteria.get_architecture())
              # Insert it at a random position in the current bacteria's architecture
              new_arch.insert(random.randrange(len(new_arch) + 1), layer_to_copy.copy()) # Insert copy
            elif dead_bacteria.get_architecture(): # If current is empty, and dead has architecture
              new_arch = dead_bacteria.get_architecture().copy()


    return BacteriaModel(bacteria.model_type, new_params, new_arch)

def calculate_similarity(bacteria1, bacteria2):
    """Calculates a similarity score between two bacteria (lower is more similar)."""
    similarity = 0
    params1 = bacteria1.get_params()
    params2 = bacteria2.get_params()

    for key in params1:
        if key in params2:
            if isinstance(params1[key], float):
                similarity += abs(params1[key] - params2[key])
            elif isinstance(params1[key], int):
                similarity += abs(params1[key] - params2[key])
            # For strings, we can treat them as different (1) or the same (0)
            elif isinstance(params1[key], str) and params1[key] != params2[key]:
                similarity += 1

    # Architecture similarity (if Keras models)
    if bacteria1.model_type == 'keras' and bacteria2.model_type == 'keras':
        arch1 = bacteria1.get_architecture()
        arch2 = bacteria2.get_architecture()
        if arch1 and arch2: # Ensure not None
          # Simple similarity: count the number of matching layers
          for i in range(min(len(arch1), len(arch2))):
              if arch1[i] == arch2[i]:
                  similarity -= 1  # Reduce similarity for matching layers
    return similarity

def calculate_diversity_penalty(population, diversity_weight=0.1):
    """Calculates a diversity penalty for each bacterium in the population."""
    for i in range(len(population)):
        total_similarity = 0
        for j in range(len(population)):
            if i != j:
                total_similarity += calculate_similarity(population[i], population[j])
        population[i].similarity_penalty = diversity_weight * total_similarity / (len(population) - 1)  # Average similarity

def adjusted_fitness(bacteria):
    """Returns the fitness adjusted for the diversity penalty."""
    return bacteria.fitness - bacteria.similarity_penalty


def run_bacteria_algorithm(X, y, model_type='xgboost', pop_size=20, generations=50,
                           conjugation_rate=0.7, transformation_rate=0.3, mutation_rate=0.1,
                           diversity_weight=0.1, n_jobs=-1):
    """Runs the bacteria-inspired algorithm."""

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    population = [BacteriaModel(model_type=model_type) for _ in range(pop_size)]
    graveyard = []

    best_fitness_history = []
    best_model = None
    best_fitness = -1

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")

        # Training and Evaluation (in parallel)
        with Parallel(n_jobs=n_jobs) as parallel:
            parallel(delayed(bac.train)(X_train, y_train, X_val, y_val) for bac in population)

        # Find and record best fitness
        for bac in population:
            if bac.fitness > best_fitness:
                best_fitness = bac.fitness
                best_model = bac
        best_fitness_history.append(best_fitness)
        print(f"  Best Fitness: {best_fitness:.4f}")

        # Calculate diversity penalties
        calculate_diversity_penalty(population, diversity_weight=diversity_weight)

        # HGT and Mutation
        new_population = []
        for i in range(0, pop_size, 2):
            bac1 = population[i]
            bac2 = population[i + 1] if i + 1 < pop_size else population[0]

            if random.random() < conjugation_rate:
                bac1, bac2 = conjugation(bac1, bac2)
            if random.random() < transformation_rate:
                bac1 = transformation(bac1, graveyard)
            if random.random() < transformation_rate:
                bac2 = transformation(bac2, graveyard)

            bac1.mutate(mutation_rate)
            bac2.mutate(mutation_rate)

            new_population.extend([bac1, bac2])

        # Selection (based on adjusted fitness)
        population.sort(key=adjusted_fitness, reverse=True)
        graveyard.extend(population[pop_size:])
        population = new_population
        population.sort(key=adjusted_fitness, reverse = True) #Sort new_population
        population = population[:pop_size] # Take best performing bacteria