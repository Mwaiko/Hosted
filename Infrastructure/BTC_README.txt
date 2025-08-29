
Hybrid Regime Pipeline Model Files
==================================

Model saved successfully with the following components:

Main Components:
- Infrastructure/BTC_main.pkl (sklearn models, scaler, regime mappings)
- Infrastructure/BTC_models.json (neural network model paths)

Neural Network Models:
- Infrastructure/BTC_regime_0.keras (regime_0)
- Infrastructure/BTC_regime_1.keras (regime_1)
- Infrastructure/BTC_regime_2.keras (regime_2)
- Infrastructure/BTC_global.keras (global)
- Infrastructure/BTC_metadata.json (training metadata)

Configuration:
- Regimes: 3 active regimes [0, 1, 2]
- Tree method: gradient_boosting
- Global fallback: Enabled

To load this model, use:
    pipeline = FixedHybridRegimePipeline.load_model('Infrastructure/BTC')
