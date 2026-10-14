## Resumen

<!-- Qué cambia y por qué. Enlaza el issue: Closes #123 -->

## Tipo de cambio

- [ ] Bug fix
- [ ] Nueva funcionalidad / herramienta
- [ ] Documentación
- [ ] Refactor / mantenimiento
- [ ] Cambio sensible (seguridad, rutas de archivos, procesos externos, herramientas de agente, dependencias, CI)

## Checklist

- [ ] `ruff format --check`, `ruff check` y `mypy reverse3d` pasan
- [ ] `pytest` pasa y añadí pruebas (positivas **y** negativas si el cambio es sensible)
- [ ] Si cambié herramientas de agente: regeneré `docs/tools/*.json`
- [ ] No hay secretos, rutas internas ni datos personales en código, tests o logs
- [ ] Nuevas dependencias justificadas en `docs/inventario-librerias.md`
- [ ] Actualicé README / docs / CHANGELOG cuando aplica

## Notas de seguridad

<!-- Para cambios sensibles: amenazas consideradas, controles añadidos, riesgos residuales. -->
