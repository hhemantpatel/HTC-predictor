import joblib
import json
import numpy as np
import os

def export_to_js():
    # Load artifacts
    model = joblib.load(os.path.join("models", "htc_model.pkl"))
    scaler = joblib.load(os.path.join("models", "scaler.pkl"))
    feature_names = joblib.load(os.path.join("models", "feature_names.pkl"))
    
    # Extract Scaler Params
    scaler_mean = scaler.mean_.tolist()
    scaler_scale = scaler.scale_.tolist()
    
    # Extract Model Weights & Biases
    # coefs_ is a list of weight matrices
    # intercepts_ is a list of bias vectors
    weights = [w.tolist() for w in model.coefs_]
    biases = [b.tolist() for b in model.intercepts_]
    
    # Activation function
    activation = model.activation
    
    # Create JS content
    js_content = f"""
// Model Artifacts
const FEATURE_NAMES = {json.dumps(feature_names)};
const SCALER_MEAN = {json.dumps(scaler_mean)};
const SCALER_SCALE = {json.dumps(scaler_scale)};
const WEIGHTS = {json.dumps(weights)};
const BIASES = {json.dumps(biases)};
const ACTIVATION = "{activation}";

// Inference Function
function predict(inputs) {{
    // 1. Preprocess (Scale)
    let scaled_inputs = inputs.map((val, idx) => {{
        return (val - SCALER_MEAN[idx]) / SCALER_SCALE[idx];
    }});
    
    // 2. Forward Pass
    let layer_input = scaled_inputs;
    
    for (let i = 0; i < WEIGHTS.length; i++) {{
        let w = WEIGHTS[i];
        let b = BIASES[i];
        
        // Matrix multiplication: input * weights + bias
        let layer_output = [];
        for (let j = 0; j < w[0].length; j++) {{
            let sum = 0;
            for (let k = 0; k < layer_input.length; k++) {{
                sum += layer_input[k] * w[k][j];
            }}
            sum += b[j];
            layer_output.push(sum);
        }}
        
        // Activation (apply to all except last layer)
        if (i < WEIGHTS.length - 1) {{
            if (ACTIVATION === 'relu') {{
                layer_output = layer_output.map(x => Math.max(0, x));
            }} else if (ACTIVATION === 'tanh') {{
                layer_output = layer_output.map(x => Math.tanh(x));
            }} else if (ACTIVATION === 'logistic') {{
                layer_output = layer_output.map(x => 1 / (1 + Math.exp(-x)));
            }}
        }}
        
        layer_input = layer_output;
    }}
    
    return layer_input[0];
}}
"""
    
    with open("model.js", "w") as f:
        f.write(js_content)
    print("Exported model.js successfully.")

if __name__ == "__main__":
    export_to_js()
