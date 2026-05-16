# Paper Signal Tester 1m

Estrategia de prueba para verificar que el IRT/paper trading abre y cierra posiciones ficticias.

No esta hecha para ganar dinero. Solo genera senales frecuentes:

- Compra en minutos divisibles por 6.
- Vende 3 minutos despues.

Uso:

1. Sube `paper_signal_tester_1m.py` desde TestBack.
2. Inicia IRT con timeframe `1m`.
3. Usa `Actualizar ahora` despues de que cierre una vela nueva.
4. En pocos minutos deberias ver BUY, posicion activa, SELL y trades cerrados.
