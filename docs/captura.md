# Guía de captura: fotos para un modelo 3D excelente

La calidad del modelo depende sobre todo de las fotos. Siguiendo esta guía, reverse3d (y reverse3d Studio) reconstruye el objeto completo, con su forma real y listo para medir o imprimir.

## ¿Cuántas fotos?

| | Fotos | Para qué |
|---|---|---|
| **Mínimo** | 20 | Objetos pequeños y sencillos, con buena textura. |
| **Recomendado** | 40–60 | La mayoría de los objetos: piezas, figuras, herramientas. |
| **Grandes o complejos** | 80–150 | Muebles, motores, objetos con huecos o mucho detalle. |

Con menos de 20 fotos la fotogrametría rara vez cierra el objeto. En ese caso el piloto automático entrega un **modelo aproximado** a partir de la mejor foto (ver más abajo).

## Los 4 perfiles de foto

Da la vuelta completa al objeto en tres alturas y termina con fotos desde arriba. El ángulo es la inclinación de la cámara respecto a la mesa.

| # | Perfil | Ángulo | Fotos | Cómo |
|---|---|---|---|---|
| 1 | Anillo bajo | 10–20° | 16 | Casi a la altura de la mesa. Una foto cada ~22°. |
| 2 | Anillo medio | 35–45° | 16 | Mirando el objeto en diagonal. Una foto cada ~22°. |
| 3 | Anillo alto | 60–70° | 12 | Desde arriba, inclinado. Una foto cada 30°. |
| 4 | Cenital | 90° | 4 | Justo encima, girando un cuarto de vuelta entre fotos. |

**Total recomendado: 48 fotos.** Para la base, voltea el objeto, repite los anillos 1 y 2 y procesa esas fotos como otro proyecto.

```
            4 (90°)
            ●
      3 ●   |   ● 3   (60–70°)
   2 ●      |      ● 2   (35–45°)
 1 ●     [objeto]     ● 1   (10–20°)
 ──────────────────────────── mesa
```

## Reglas para un resultado excelente

1. **Solape del 60–80 %.** Cada foto debe mostrar casi lo mismo que la anterior. Muévete poco entre una y otra, siempre en orden.
2. **El objeto no se mueve.** Camina tú alrededor; no lo gires, no lo levantes ni lo cambies de sitio entre fotos.
3. **Luz pareja y difusa.** Luz de día indirecta, día nublado o una lámpara rebotada en la pared. Sin flash ni sol directo. Apaga luces RGB o pantallas que cambien de color.
4. **Nada brillante ni transparente.** Cromados, plásticos brillantes, negro brillante, vidrio o espejos confunden a la cámara: cúbrelos con spray mate para escaneo, talco o champú seco.
5. **Encuadre.** El objeto entero dentro de la foto, ocupando el 50–70 % del encuadre. El mismo zoom en todas; nada de zoom digital.
6. **Fotos nítidas.** Enfoca el objeto y mantén el teléfono quieto. Desactiva el modo retrato, los filtros y el HDR extremo.
7. **Fondo con textura.** Apoya el objeto sobre un periódico, una alfombrilla con dibujo o una mesa con vetas: ayuda a ubicar cada foto.
8. **Medidas reales.** Pon una regla o un objeto de tamaño conocido al lado y ajusta la escala después (`--scale`).

**Formatos:** JPG, PNG, HEIC (iPhone), TIFF y WebP. Mínimo 8 MP: la cámara de cualquier teléfono actual sirve.

## Qué hace reverse3d si las fotos no cumplen la guía

1. **Compara todas las fotos entre sí** (no importa el orden) y busca el grupo que se solapa.
2. **Si una parte se solapa (8 o más fotos)**, reconstruye con esa parte y avisa de cuántas omitió.
3. **Si COLMAP no logra una reconstrucción suficiente** (menos de 5 fotos ubicadas o menos de 1.000 puntos 3D), no pierde tiempo en la fase densa.
4. **Plan B:** elige la foto que mejor muestra el objeto, lo recorta del fondo y genera un **sólido cerrado aproximado** con IA (`fallback` en el resumen), con estos consejos para repetir la captura.

El modelo aproximado tiene la cara oculta plana y la altura estimada. Para el objeto completo y con medidas reales, sigue esta guía.