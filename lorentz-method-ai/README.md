# Lorentz Method AI

This folder contains TradingView Pine Script indicators that implement a Lorentzian-distance-based nearest neighbors classifier. The primary script is **Machine Learning Lorentzian Classification.ps**, which builds a small ML-style pipeline inside Pine Script to predict the next four bars of price direction and generate trade signals.

## Full outline of `Machine Learning Lorentzian Classification.ps`

1. **Header + Imports**
   - Declares the indicator and imports shared modules for ML utilities and kernel regression.
2. **Background Commentary**
   - Explains why Lorentzian distance can be a better metric for market data vs. Euclidean distance and sets expectations for feature behavior.
3. **Custom Types**
   - Defines structured data types for settings, labels, features, model state, and filters.
4. **Helper Functions**
   - `series_from(...)`: maps a feature name (RSI/WT/CCI/ADX) to a normalized series.
   - `get_lorentzian_distance(...)`: computes Lorentzian distance across 2–5 features.
5. **Inputs + Settings**
   - General settings (source, neighbors, bars back, feature count, color compression, exits).
   - Filter settings (volatility, regime, ADX, EMA/SMA).
   - Feature engineering inputs (feature selection + parameters for 5 feature slots).
   - Kernel regression inputs (lookback, weighting, regression level, lag, smoothing options).
   - Display inputs (bar colors, prediction labels, offsets).
6. **Feature Engineering**
   - Computes per-bar feature series from user-selected feature types.
   - Stores rolling history in arrays to feed ML distance calculations.
7. **Training Label Construction**
   - Builds a label per bar based on direction over the next 4 bars (long/short/neutral).
8. **Core ML Logic**
   - Runs an approximate nearest neighbors loop with Lorentzian distance.
   - Maintains `neighborsCount` predictions and sums them into a final `prediction` score.
9. **Prediction Filters**
   - Combines volatility/regime/ADX filters with EMA/SMA trend filters.
   - Applies bar-count and fractal-based filters for signal stability.
10. **Kernel Regression Logic**
    - Computes a Nadaraya-Watson estimate with Rational Quadratic + Gaussian kernels.
    - Generates bullish/bearish regime filters and crossover/change alerts.
11. **Entries and Exits**
    - Entry signals require a new ML signal plus optional kernel + trend filters.
    - Exits can be strict (4-bar holding period) or dynamic (kernel change).
12. **Plotting + Alerts**
    - Plots entry/exit labels and kernel estimate.
    - Defines alert conditions for entries, exits, and kernel changes.
13. **Backtesting Helpers**
    - Streams signals for external backtest adapters.
    - Displays summary trade stats in a table when enabled.

## 1. Logic to run the script

1. **Add to TradingView**
   - Open TradingView’s Pine Editor and paste the contents of `Machine Learning Lorentzian Classification.ps`.
   - Save and add to a chart.
2. **Configure General Settings**
   - Choose the input `Source`, neighbor count, max bars back, and feature count.
   - Decide if you want default exits or dynamic exits.
3. **Configure Filters**
   - Toggle volatility/regime/ADX filters to reduce noise.
   - Optional EMA/SMA trend filters enforce trading only in aligned trends.
4. **Feature Engineering**
   - Select 2–5 features (RSI/WT/CCI/ADX) and their parameters.
5. **Kernel Settings**
   - Enable kernel filtering and tuning for smoothing or crossover-based signals.
6. **Visual + Stats**
   - Toggle bar colors, prediction labels, and trade stats table.

## 2. Calculations behind the script

### Feature generation
- The script computes up to five normalized features via `series_from(...)`:
  - RSI, WT (WaveTrend), CCI, or ADX from the `MLExtensions` library.
- Each feature series is appended to a rolling array for distance calculations.

### Training labels (supervised signal)
- The script uses a **fixed 4-bar prediction horizon**:
  - If `src[4] < src[0]`, label is **short**.
  - If `src[4] > src[0]`, label is **long**.
  - Else label is **neutral**.

### Lorentzian distance
- Distance between current feature vector and a historical vector uses:
  - `log(1 + abs(feature_i - historical_feature_i))` summed across 2–5 features.
- This reduces the impact of outliers and warps extreme distances similar to a “price-time” analogy.

### Approximate nearest neighbors (ANN)
- Iterates through historical bars (with a modulo spacing of 4 bars) and:
  - Computes Lorentzian distance.
  - Maintains the most recent `neighborsCount` predictions + distances.
  - Uses a distance threshold from the lower 25% to throttle neighbor growth.
- Final prediction = sum of neighbor labels (positive = long bias, negative = short bias).

### Filters
- **Volatility filter**: reduces signals in low-volatility regimes.
- **Regime filter**: attempts to avoid ranging markets using a threshold.
- **ADX filter**: optionally requires a minimum trend strength.
- **EMA/SMA filters**: optional trend confirmation.
- **Kernel regression filter**: uses Nadaraya-Watson estimate for bullish/bearish regime detection.

### Kernel regression
- `kernels.rationalQuadratic(...)` and `kernels.gaussian(...)` create a smoothed price estimate.
- The script tracks:
  - Rate-of-change based regime (bullish/bearish).
  - Crossovers between kernels for smoother transitions (if enabled).

## 3. Logic to go long or short

### Long entries
A long trade can start when:
- The ML signal flips to a **new long** (`isNewBuySignal`).
- Kernel filter is bullish (if enabled).
- EMA/SMA filters show uptrend (if enabled).

### Short entries
A short trade can start when:
- The ML signal flips to a **new short** (`isNewSellSignal`).
- Kernel filter is bearish (if enabled).
- EMA/SMA filters show downtrend (if enabled).

### Exits
- **Strict exits** (default):
  - Exit after 4 bars, or if a new opposite signal appears before 4 bars (with conditions).
- **Dynamic exits** (optional):
  - Exit on kernel regime flips, provided EMA/SMA filters are off and kernel smoothing is disabled.

## Files in this folder

- `Machine Learning Lorentzian Classification.ps`: Main indicator with ML classification + entry/exit logic.
- `Lorentzian-KNN-strategy-AI.ps`: Related strategy script.
- `MLExtensions.ps`, `KernelFunctions.ps`: Shared libraries.
- `Backtest Adapter.ps`: Backtest adapter for signal streaming.
