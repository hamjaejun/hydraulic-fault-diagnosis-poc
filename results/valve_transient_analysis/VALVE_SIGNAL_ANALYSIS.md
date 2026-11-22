# Valve Signal Analysis

## Scope and interpretation

The UCI documentation states that the rig repeats fixed 60-second load cycles, but it does not publish a valve command trace or spool position. Therefore the event below is a **measured sensor-response event**, not a directly measured switching command.

## Observed first response

Across the class-mean PS1–PS3, FS1, and EPS1 signals, the strongest early derivative occurs around 9.3–10.2 seconds. The class-specific EPS1 peaks range from 9.8 to 10.2 seconds. This independently reproduces the approximate region reported by the teammate, without treating their accumulator work as part of this repository.

The 0–5 second plots show far less separation because the dominant pressure/flow/power response has not yet occurred. This is evidence consistent with insufficient early-cycle switching information, not proof of the physical command timing.

## Feature policy

Transient features are computed from each cycle independently. Event-specific pre/post, t10, t90, rise-time, overshoot, and settling proxies are created only for observation windows of at least 12 seconds. No future sample is used, and no pressure-pair feature is called a valve pressure drop because sensor placement is not available in the dataset documentation.

## Figures

- `valve_class_mean_signals.png`: full-cycle class means for nine pressure/flow/power sensors.
- `initial_5s_class_means.png`: initial region before the dominant measured response.
- `first_response_class_means.png`: first measured response region with the observed 9.3–10.2 s band.
- `individual_cycle_variability.png`: individual cycles plus class means.
- `cooler_stratified_response.png`: same valve class under different cooler domains.