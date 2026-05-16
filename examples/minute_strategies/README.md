# Minute Strategies

Estrategias experimentales para probar IRT en timeframes pequenos.

## btc_1m_momentum_pulse.py

Estrategia long-only para BTC/USDT 1m:

- Compra con tendencia corta alineada.
- Exige ruptura de maximo reciente.
- Filtra RSI, volumen y volatilidad.
- Sale por take-profit, stop-loss, trailing stop, fallo de tendencia o tiempo.

En 1m las comisiones pesan mucho. Pruebala primero en TestBack y despues en IRT con dinero ficticio.
