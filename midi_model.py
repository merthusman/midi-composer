# src/model/midi_model.py (TensorFlow/Keras Based Model)
import tensorflow as tf
import numpy as np
import os
import random
from typing import List, Optional, Dict, Tuple, Any
import logging

# Keras imports
from tensorflow.keras.layers import (
    LSTM, Dense, Input, Dropout, Layer,
    MultiHeadAttention, LayerNormalization,
    Reshape, TimeDistributed, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Use tf.math instead of K
import tensorflow.math as math

# Local imports
try:
    from src.core.settings import Settings, ModelSettings
    _settings_imported_in_model = True
except ImportError as e:
    logging.error(f"Failed to import Settings in midi_model: {e}")
    _settings_imported_in_model = False
    
    class Settings:
        def __init__(self):
            self.model_settings = ModelSettings()
            
    @dataclass
    class ModelSettings:
        sequence_length: int = 32
        note_range: Tuple[int, int] = (21, 108)
        input_features: int = 2  # pitch and velocity
        output_features: int = 2  # match input features
        resolution: float = 0.125  # 1/8th note resolution
        lstm_units: int = 128
        dense_units: int = 128
        dropout_rate: float = 0.3
        batch_size: int = 32
        learning_rate: float = 0.001


logger = logging.getLogger(__name__) # Get logger for this module


# --- Custom Layers and Loss Function ---

# Using tf.keras.losses.Loss class instead of a standalone function for the loss
class CustomCombinedLoss(tf.keras.losses.Loss):
    """Custom combined loss function for MIDI generation.
    Can combine pitch loss, rhythm loss, etc.
    Example: Binary Crossentropy for pitch activation, MSE for velocity/duration (if included).
    """
    def __init__(self, pitch_loss_weight=1.0, rhythm_loss_weight=0.5, name="custom_combined_loss"):
        super().__init__(name=name)
        self.pitch_loss_weight = pitch_loss_weight
        self.rhythm_loss_weight = rhythm_loss_weight # Weight for rhythm loss if implemented
        
        # Define individual loss functions
        self.pitch_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False) # Use from_logits=False if model output is sigmoid activated
        
    def call(self, y_true, y_pred):
        """
        Calculates the combined loss.
        y_true and y_pred should have the same shape: (batch_size, sequence_length, note_range_size, output_features)
        Assuming output_features=1: last dimension is pitch activation (sigmoid)
        Assuming output_features=2: last dimension is pitch activation (sigmoid), velocity (linear)
        """
        # Split y_true and y_pred into pitch and rhythm components
        y_true_pitch = y_true[..., 0]  # First feature is pitch
        y_pred_pitch = y_pred[..., 0]  # First feature is pitch
        
        # Calculate pitch loss
        pitch_loss = self.pitch_loss_fn(y_true_pitch, y_pred_pitch)
        
        # If we have velocity/rhythm data (output_features > 1)
        if y_true.shape[-1] > 1:
            y_true_rhythm = y_true[..., 1]  # Second feature is velocity/rhythm
            y_pred_rhythm = y_pred[..., 1]  # Second feature is velocity/rhythm
            
            # Calculate rhythm loss (using MSE for velocity)
            rhythm_loss = tf.keras.losses.mean_squared_error(y_true_rhythm, y_pred_rhythm)
            
            # Combine losses
            return self.pitch_loss_weight * pitch_loss + self.rhythm_loss_weight * rhythm_loss
        
        # If only pitch data
        return pitch_loss



# Custom Metric: Binary Accuracy for Pitch Activation
class PitchBinaryAccuracy(tf.keras.metrics.Metric):
    """
    Calculates binary accuracy specifically for the pitch activation feature (output_features=1).
    Assumes output_features=1 and uses a threshold (e.g., 0.5) for binary prediction.
    """
    def __init__(self, threshold=0.5, name='pitch_binary_accuracy', dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.threshold = threshold
        # Use tf.math instead of K
        self._binary_accuracy = tf.keras.metrics.BinaryAccuracy(threshold=self.threshold)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Extract pitch activation from y_true and y_pred
        y_true_pitch = y_true[..., 0]  # First feature is pitch
        y_pred_pitch = y_pred[..., 0]  # First feature is pitch
        
        # Update the internal binary accuracy metric
        self._binary_accuracy.update_state(y_true_pitch, y_pred_pitch, sample_weight)
        
    def result(self):
        """Returns the metric result."""
        return self._binary_accuracy.result()

    def reset_state(self):
        """Resets the state of the metric."""
        self._binary_accuracy.reset_state()

    # Optionally, add from_config and get_config for serialization if needed
    # def get_config(self):
    #     config = super().get_config()
    #     config.update({"threshold": self.threshold})
    #     return config
    #
    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)


# Custom Layer: Style/Repeat Layer (Placeholder - needs implementation)
class StyleRepeatLayer(Layer):
    """
    A custom layer that could incorporate style embeddings or learn to repeat patterns.
    This is a placeholder and needs significant implementation based on your design.
    Example: Could take a style vector as input and modulate the main sequence processing.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize weights/variables if the layer has trainable parameters

    def build(self, input_shape):
        # Define the shape of the layer's weights based on input_shape
        # Example: self.kernel = self.add_weight(...)
        super().build(input_shape)

    def call(self, inputs):
        # Define the layer's forward pass logic
        # Example: return inputs * self.kernel
        # This layer likely needs multiple inputs: the main sequence data and perhaps a style vector
        return inputs # Placeholder: simply pass inputs through for now

    def get_config(self):
        # Implement this if you need to save/load models containing this layer
        config = super().get_config()
        # Add custom configuration parameters here
        return config


# Custom Layer: Transformer Encoder Block 
class TransformerEncoderBlock(Layer):
    """
    A basic Transformer Encoder block.
    Includes Multi-Head Self-Attention, Add & Norm, and Feed Forward.
    
    Args:
        embed_dim: Dimension of the embedding space
        num_heads: Number of attention heads
        ff_dim: Hidden layer size in feed forward network
        rate: Dropout rate
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        # Initialize layers
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        """
        Forward pass of the Transformer Encoder block.
        
        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, embed_dim)
            training: Boolean indicating whether the layer is in training mode
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, embed_dim)
        """
        # Multi-head self-attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward neural network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# --- MIDI Model Class ---

class MIDIModel:
    # Pass settings object to the model
    # Removed individual settings parameters like note_range_size, steps_per_bar etc.
    def __init__(self, settings): # Now accepts a settings object
        if not _settings_imported_in_model:
            logger.error("Settings dataclass not imported. Cannot initialize MIDIModel.")
            raise ImportError("Settings dataclass is required for MIDIModel initialization.")

        self.settings = settings # Store the settings object
        # Access model settings from the settings object
        self.sequence_length = self.settings.model_settings.sequence_length
        self.note_range = self.settings.model_settings.note_range
        self.input_features = self.settings.model_settings.input_features
        self.output_features = self.settings.model_settings.output_features # Should be same as input_features for training? Or 1 (pitch)?
        self.lstm_units = self.settings.model_settings.lstm_units
        self.dense_units = self.settings.model_settings.dense_units
        self.dropout_rate = self.settings.model_settings.dropout_rate
        self.batch_size = self.settings.model_settings.batch_size
        self.learning_rate = self.settings.model_settings.learning_rate
        self.resolution = self.settings.model_settings.resolution

        # Calculate note_range_size and steps_per_bar based on settings
        self.note_range_size = self.note_range[1] - self.note_range[0] + 1
        # Assuming 4/4 time signature for steps_per_bar calculation based on resolution
        self.steps_per_bar = int(4 / self.resolution) # 4 quarter notes * steps per quarter note

        # Update settings for input/output features based on actual data
        self.input_features = 2  # We have 2 features: pitch and velocity
        self.output_features = 2  # Output should match input features

        self.model = None # Keras model object will be stored here
        self._is_trained = False # Flag to indicate if the model has been trained

        logger.debug(f"MIDIModel initialized with resolution={self.resolution}, calculated steps_per_bar = {self.steps_per_bar}, note_range_size = {self.note_range_size}")
        # Note: Model is NOT built/compiled in __init__. Call build_model separately.


    def build_model(self):
        """Builds and compiles the Keras model."""
        logger.info("Building Keras model...")
        # Define the input shape: (sequence_length, note_range_size, input_features)
        # Batch size is handled by Keras implicitly or can be specified as the first dim if stateful=True for LSTM
        input_shape = (self.sequence_length, self.note_range_size, self.input_features)
        logger.debug(f"Model input shape: {input_shape}")

        # --- Model Architecture Definition ---
        # Input layer: Specify the shape *excluding* the batch size
        input_layer = Input(shape=(self.sequence_length, self.note_range_size, self.input_features), name='input_sequence')

        # Reshape to flatten the note_range_size and input_features into one dimension for LSTM input
        # LSTM expects input shape (batch_size, sequence_length, features)
        # New shape will be (batch_size, sequence_length, note_range_size * input_features)
        reshaped_input = Reshape((self.sequence_length, self.note_range_size * self.input_features))(input_layer)
        logger.debug(f"Reshaped input shape for LSTM: {reshaped_input.shape}")


        # LSTM layers
        # return_sequences=True is needed to output a sequence for the next LSTM or TimeDistributed layer
        # If using multiple LSTMs stacked, all but the last should have return_sequences=True
        # If using a final Dense layer directly, the last LSTM should have return_sequences=False (or handle outputs accordingly)
        # For sequence generation, we need output for each time step, so return_sequences=True for the last LSTM is likely correct if followed by TimeDistributed
        lstm_out = LSTM(self.lstm_units, return_sequences=True, name='lstm_layer_1')(reshaped_input)
        lstm_out = Dropout(self.dropout_rate)(lstm_out) # Apply dropout after LSTM

        # Add more LSTM layers if needed
        # lstm_out = LSTM(self.lstm_units, return_sequences=True, name='lstm_layer_2')(lstm_out)
        # lstm_out = Dropout(self.dropout_rate)(lstm_out)


        # Apply a TimeDistributed Dense layer to get output for each step
        # This layer applies the same Dense operation to each time step independently.
        # Output shape from TimeDistributed(Dense) will be (batch_size, sequence_length, dense_output_units)
        # We want output shape (batch_size, sequence_length, note_range_size, output_features)
        # So, the final Dense layer needs to output note_range_size * output_features units, and then we reshape.
        dense_out = TimeDistributed(Dense(self.note_range_size * self.output_features, activation='sigmoid'), name='time_distributed_dense')(lstm_out)
        # Using sigmoid activation for the final output if output_features=1 (pitch activation probabilities)
        # If output_features=2 (pitch, velocity), you might need different activations and potentially two separate Dense layers or careful handling.
        # Assuming output_features=1 for simplicity in activation choice here.

        # Reshape the output back to (batch_size, sequence_length, note_range_size, output_features)
        output_layer = Reshape((self.sequence_length, self.note_range_size, self.output_features), name='output_sequence')(dense_out)
        logger.debug(f"Model output shape: {output_layer.shape}")


        # --- Model Creation ---
        self.model = Model(inputs=input_layer, outputs=output_layer, name='midi_generator_model')

        # --- Model Compilation ---
        # Use the custom combined loss
        # Pass custom_objects to compile if custom layers/losses are used in the model definition itself (not just loading)
        # Our custom loss is defined as a class inheriting from tf.keras.losses.Loss, so Keras recognizes it.
        # However, if the model structure included custom *layers* like StyleRepeatLayer or TransformerEncoderBlock,
        # you would need to potentially pass custom_objects={...} to tf.keras.models.load_model.
        # For compilation, the custom loss instance is enough.
        custom_loss_instance = CustomCombinedLoss(pitch_loss_weight=1.0, rhythm_loss_weight=0.0) # Adjust weights as needed
        optimizer = Adam(learning_rate=self.learning_rate) # Use the learning rate from settings

        # Define metrics - using the custom pitch binary accuracy
        metrics = [PitchBinaryAccuracy(name='pitch_acc')] # Add other metrics if needed

        self.model.compile(optimizer=optimizer, loss=custom_loss_instance, metrics=metrics) # Use the custom loss instance


        logger.info("Keras model built and compiled successfully.")
        self.model.summary(print_fn=logger.info) # Print model summary to log


    def train(self, training_data: np.ndarray, epochs: int = 10, validation_data: Optional[np.ndarray] = None, callbacks: Optional[List[tf.keras.callbacks.Callback]] = None):
        """
        Trains the model.
        training_data shape: (num_samples, sequence_length, note_range_size, input_features)
        """
        if self.model is None:
            logger.error("Model is not built yet. Call build_model() before training.")
            # Decide how to handle: raise error, return False, build automatically?
            self.build_model()  # Auto-build for convenience if not built

        logger.info(f"Starting model training for {epochs} epochs...")
        try:
            # Prepare data for training
            # Assuming training_data is already batched or will be handled by Keras fit
            # For this model, input and output (target) should be the same sequence data for learning to predict the next step/sequence.
            # Training setup depends on the exact task:
            # - Predict next step given sequence (many-to-one logic at the end)
            # - Predict entire output sequence given input sequence (sequence-to-sequence)
            # Based on the LSTM return_sequences=True and TimeDistributed(Dense), it seems to be sequence-to-sequence where we predict the next 'frame' for each step.
            # So, y_true should be the same as the input data shifted by one step, or the input data itself if predicting the next state at each step.
            # A common approach for sequence generation training is to use the input sequence as features and the *same* sequence (or a shifted version) as labels.
            # For this model's architecture (predicting output_features for each step), y_true = training_data is a valid approach if learning to reconstruct/continue the sequence frame by frame.
            # If predicting the NEXT frame, you need to shift the data: X = data[:, :-1, ...], y = data[:, 1:, ...]

            # Verify and prepare training data shape
            if training_data.ndim == 4:
                # Expected shape: (batch_size, sequence_length, note_range_size, input_features)
                X_train = training_data
                y_train = training_data
                logger.info(f"Training data shape: {X_train.shape}")
            else:
                logger.error(f"Invalid training data shape: {training_data.shape}. Expected (batch_size, sequence_length, note_range_size, input_features).")
                return None

            # Update batch size to match actual data
            self.batch_size = X_train.shape[0]
            logger.info(f"Updated batch size to: {self.batch_size}")

            # Verify dimensions match settings
            expected_shape = (self.batch_size, self.sequence_length, self.note_range_size, self.input_features)
            if X_train.shape != expected_shape:
                logger.error(f"Training data shape {X_train.shape} does not match expected shape {expected_shape}")
                return None

            # Let's assume the model predicts the frame at time t given features up to time t.
            # In that case, y_train is simply the input data.

            # If using validation data, prepare it similarly
            if validation_data is not None:
                if validation_data.ndim == 4:
                    X_val = validation_data
                    y_val = validation_data  # Assuming same logic for validation
                elif validation_data.ndim == 3:
                    # Add a batch dimension
                    X_val = np.expand_dims(validation_data, axis=0)
                    y_val = np.expand_dims(validation_data, axis=0)
                    logger.info(f"Added batch dimension to validation data. New shape: {X_val.shape}")
                else:
                    logger.error(f"Invalid validation data shape: {validation_data.shape}. Expected (num_samples, sequence_length, note_range_size, input_features) or (sequence_length, note_range_size, input_features).")
                    X_val = None
                    y_val = None
            else:
                X_val = None
                y_val = None

            # Use Keras Model.fit for training
            try:
                history = self.model.fit(
                    X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=self.batch_size,
                    validation_data=(X_val, y_val) if validation_data is not None else None,
                    callbacks=callbacks
                )
                self._is_trained = True
                logger.info("Model training finished.")
                return history
            except Exception as e:
                logger.error(f"Model training error: {e}", exc_info=True)
                return None

        except Exception as e:
            logger.error(f"Error during model training: {e}", exc_info=True)
            return None  # Return None on error


    def generate_sequence(self, seed_sequence: np.ndarray, num_steps_to_generate: int, temperature: float = 1.0) -> Optional[np.ndarray]:
        """
        Generates a new sequence based on a seed sequence.
        seed_sequence shape: (1, sequence_length, note_range_size, input_features) - batch size 1 for generation
        Returns generated sequence shape: (num_steps_to_generate, note_range_size, output_features)
        """
        if self.model is None:
            logger.error("Model is not built or loaded yet. Cannot generate sequence.")
            return None
        if not self._is_trained:
             logger.warning("Model has not been trained. Generation may not be meaningful.")


        logger.info(f"Starting sequence generation for {num_steps_to_generate} steps...")
        try:
            # Ensure seed_sequence has correct shape (batch_size=1, sequence_length, ...)
            if seed_sequence.shape[0] != 1 or seed_sequence.shape[1] != self.sequence_length or seed_sequence.shape[2] != self.note_range_size or seed_sequence.shape[3] != self.input_features:
                logger.error(f"Seed sequence has incorrect shape: {seed_sequence.shape}. Expected (1, {self.sequence_length}, {self.note_range_size}, {self.input_features}).")
                return None

            generated_sequence = []
            current_sequence = np.copy(seed_sequence) # Start with the seed

            for _ in range(num_steps_to_generate):
                # Predict the next step based on the current sequence
                # Model input needs batch size 1
                prediction = self.model.predict(current_sequence, verbose=0) # Predict the next frame(s)

                # Process the prediction for the *next* step (the last step of the prediction output)
                # Assuming prediction shape is (1, sequence_length, note_range_size, output_features)
                # We are interested in the prediction for the frame *after* the last frame of the input sequence.
                # With return_sequences=True, the model predicts an output sequence of the same length as input.
                # The prediction at index 'i' can be interpreted as the prediction for the time step 'i' *given* the sequence up to 'i'.
                # So, to generate the next frame, we use the prediction at the *last* index of the output sequence.
                next_step_prediction = prediction[0, -1, :, :] # Shape (note_range_size, output_features)

                # Apply temperature and sampling (example for pitch activation feature 0)
                # Using softmax or sigmoid output for pitch activation
                # Assuming output_features=1 for simplicity in sampling logic
                if self.output_features >= 1:
                     pitch_probabilities = next_step_prediction[:, 0] # Get pitch probabilities (shape: note_range_size)
                     # Apply temperature (softening or sharpening probabilities)
                     # Avoid log(0) if probabilities are exactly 0
                     pitch_probabilities = np.maximum(pitch_probabilities, 1e-8) # Clip to avoid log(0)
                     pitched_probabilities = np.exp(np.log(pitch_probabilities) / temperature) # Apply temperature
                     pitched_probabilities /= np.sum(pitched_probabilities) # Re-normalize

                     # Sample the pitch activation for the next step
                     # Create the next step array (note_range_size, output_features)
                     next_step_features = np.zeros((self.note_range_size, self.output_features), dtype=np.float32)

                     # Simple binary sampling for pitch activation (feature 0) based on pitched probabilities
                     # You might want to sample multiple simultaneous notes based on probabilities
                     # For a simple approach, decide whether each note is on or off based on its probability
                     for note_idx in range(self.note_range_size):
                          if random.random() < pitched_probabilities[note_idx]: # Sample based on probability
                               next_step_features[note_idx, 0] = 1.0 # Note is active
                               # If output_features >= 2, also sample/predict velocity for active notes
                               # This needs a prediction for velocity (e.g., from feature 1 in next_step_prediction)
                               if self.output_features >= 2:
                                   # Example: Use the predicted velocity directly (scaled) or sample around it
                                   predicted_velocity = next_step_prediction[note_idx, 1] # Assuming velocity is feature 1
                                   next_step_features[note_idx, 1] = predicted_velocity # Store velocity

                # Add the sampled next step features to the generated sequence
                generated_sequence.append(next_step_features)

                # Update current_sequence by removing the first step and adding the generated step
                # This slides the sequence window forward by one step
                current_sequence = np.roll(current_sequence, shift=-1, axis=1) # Shift left by one step
                current_sequence[0, -1, :, :] = next_step_features # Replace the last step with the new generated step

            # Concatenate the generated steps into a single sequence array
            generated_sequence_array = np.array(generated_sequence) # Shape (num_steps_to_generate, note_range_size, output_features)
            logger.info(f"Sequence generation finished. Generated shape: {generated_sequence_array.shape}")
            return generated_sequence_array

        except Exception as e:
            logger.error(f"Error during sequence generation: {e}", exc_info=True)
            return None


    def save_model(self, file_path: str) -> bool:
        """Saves the trained model to a file."""
        if self.model is None or not self._is_trained:
            logger.warning("No trained model to save.")
            # Use tf.print to ensure message appears even if standard logging is misconfigured early
            # tf.print("WARNING (save): No trained model to save.", output_stream=sys.stderr) # Using logger is preferred if configured
            return False

        # Create directory if it doesn't exist
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created model save directory: {output_dir}")


        logger.info(f"Saving model to {file_path}...")
        try:
            # Define custom objects if your model uses them (custom layers, losses, metrics)
            # Ensure custom layers are defined and imported where load_model is called.
            # Here, for saving, Keras needs to know how to serialize them.
            # If you used functional API or Sequential API with string names for layers,
            # Keras might save the config, but custom layers need to be registered.
            # Passing custom_objects here explicitly ensures they are linked.
            custom_objects = {
                "CustomCombinedLoss": CustomCombinedLoss,
                "PitchBinaryAccuracy": PitchBinaryAccuracy,
                "StyleRepeatLayer": StyleRepeatLayer, # Include if used in the model architecture
                "TransformerEncoderBlock": TransformerEncoderBlock, # Include if used
                # Add other custom objects
            }

            # Use the Keras method to save the entire model (architecture, weights, optimizer state)
            self.model.save(file_path, include_optimizer=True, save_format='h5') # Or 'tf' for TensorFlow SavedModel format
            logger.info("Model saved successfully.")
            return True

        except Exception as e:
            logger.error(f"Error saving model to {file_path}: {e}", exc_info=True)
            return False

    def load_model(self, file_path: str) -> bool:
        """Loads a trained model from a file."""
        if not os.path.exists(file_path):
            logger.warning(f"Model file not found at {file_path}. Cannot load model.")
            # tf.print(f"WARNING (load): Model file not found at {file_path}.", output_stream=sys.stderr) # Using logger is preferred
            return False

        logger.info(f"Loading model from {file_path}...")
        try:
            # Define custom objects dictionary matching the ones used in the model architecture/compilation
            # This is crucial for Keras to correctly load custom components.
            custom_objects = {
                "CustomCombinedLoss": CustomCombinedLoss,
                "PitchBinaryAccuracy": PitchBinaryAccuracy,
                 # Include custom layers if they are part of the saved model architecture
                 "StyleRepeatLayer": StyleRepeatLayer,
                 "TransformerEncoderBlock": TransformerEncoderBlock,
                # Add other custom objects
            }

            # Use the Keras method to load the model
            # Pass custom_objects to the load_model function
            self.model = tf.keras.models.load_model(file_path, custom_objects=custom_objects)
            self._is_trained = True # Assume loaded model is trained
            logger.info("Model loaded successfully.")
            # Optionally compile again after loading if optimizer state wasn't saved or format requires it
            # self.model.compile(...) # Recompile if needed

            return True

        except Exception as e:
            logger.error(f"Error loading model from {file_path}: {e}", exc_info=True)
            self.model = None # Ensure model is None if loading fails
            self._is_trained = False
            return False


    # --- Prediction / Generation Helper (moved generation logic to generate_sequence) ---
    # The predict method is part of the Keras model object (self.model.predict)
    # You would use self.model.predict(...) directly for inference on data.


# Example usage (if run directly for testing)
if __name__ == "__main__":
    # Configure logging only if running THIS file directly for testing
    # In a real application, logging is configured at the main entry point
    if not logging.getLogger('').handlers: # Check if root logger has handlers
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s %(funcName)s: %(message)s")

    logger.info("Running midi_model.py test.")

    # Create a dummy settings object for testing purposes
    # In main application, a real Settings object will be passed
    @dataclass
    class DummyModelSettings(ModelSettings): # Inherit to get defaults
         # Override defaults or add specific test settings if needed
         pass # Use default model settings

    @dataclass
    class DummyGeneralSettings(GeneralSettings):
         pass # Use default general settings

    @dataclass
    class DummyMemorySettings(MemorySettings):
         pass # Use default memory settings


    @dataclass
    class DummySettings(Settings): # Inherit from main Settings
         model_settings: DummyModelSettings = field(default_factory=DummyModelSettings)
         general_settings: DummyGeneralSettings = field(default_factory=DummyGeneralSettings)
         memory_settings: DummyMemorySettings = field(default_factory=DummyMemorySettings)
         # Add _get_project_root and property methods if needed for testing file paths
         def _get_project_root(self):
              # For dummy test, use current directory as project root
              return os.path.abspath(os.path.dirname(__file__)) # This puts test files in src/model

         # Need to redefine properties that call _get_project_root if testing save/load paths
         @property
         def model_dir_path(self):
              return os.path.join(self._get_project_root(), self.general_settings.model_dir)


    dummy_settings = DummySettings()
    logger.info(f"Dummy model save path: {dummy_settings.model_dir_path}") # Test path

    # Initialize model with dummy settings
    midi_model = MIDIModel(settings=dummy_settings)

    # Test building and compiling the model
    midi_model.build_model()

    # Test saving the model (will save to a dummy
