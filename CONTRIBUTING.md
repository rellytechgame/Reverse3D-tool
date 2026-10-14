# Guía de contribución

¡Gracias por tu interés en **reverse3d**! Este proyecto es open source (Apache-2.0) y lo mantiene [Voxel Enterprise](https://voxel.lat).

## Entorno de desarrollo

```bash
git clone https://github.com/voxelenterprise/reverse3d.git
cd reverse3d
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pip install pre-commit && pre-commit install
```

## Flujo de trabajo

1. Abre un issue antes de cambios grandes para acordar el diseño.
2. Crea una rama desde `main`: `feat/...`, `fix/...`, `docs/...`. No se hace push directo a `main`.
3. Usa commits pequeños con mensajes en formato [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
4. Abre un PR usando la plantilla. Debe pasar CI y tener al menos una aprobación de CODEOWNER.

## Estándares de código

- Python ≥ 3.10, con type hints en toda la API pública (`mypy reverse3d` sin errores).
- `ruff format` y `ruff check` sin errores; el límite de línea es 100.
- Toda función pública lleva docstring: propósito, argumentos, retorno y errores (`Raises:`).
- Los comentarios explican **por qué**, no qué.
- Los errores esperados se lanzan como subclases de `Reverse3DError`, con un `code` estable y un `hint` accionable.

## Reglas de seguridad (obligatorias)

- **Nunca** uses `shell=True`, `eval`, `exec`, `pickle`, `allow_pickle=True` ni `trust_remote_code=True`.
- Los procesos externos se lanzan **solo** con `reverse3d.security.run_command`.
- Todo archivo de entrada pasa por `validate_input_file`, con lista blanca de extensiones y límite de tamaño. Toda salida pasa por `prepare_output_file`.
- Las nuevas herramientas para agentes deben:
  - usar un modelo Pydantic con `extra="forbid"` y límites en todos los números;
  - resolver **todas** las rutas con `resolve_in_workspace`;
  - no sobrescribir archivos salvo con `overwrite=true` explícito;
  - incluir tests positivos **y** negativos (traversal, argumentos extra, límites);
  - regenerar `docs/tools/*.json` con `reverse3d agent tools --format <dialect> -o docs/tools/<dialect>.json`.
- No incluyas secretos, rutas internas ni datos personales en código, tests, fixtures o logs.
- Cada dependencia nueva se justifica en [docs/inventario-librerias.md](docs/inventario-librerias.md).

## Tests

```bash
pytest --cov=reverse3d          # cobertura mínima: 70 %
ruff format --check reverse3d tests && ruff check reverse3d tests && mypy reverse3d
bandit -c pyproject.toml -r reverse3d && pip-audit --skip-editable
```

Los tests no requieren COLMAP ni Blender: los procesos externos se simulan. Los tests de integración con binarios reales llevan el marcador `@pytest.mark.integration`.

## Código de conducta

Este proyecto sigue el [Código de Conducta](CODE_OF_CONDUCT.md). Al participar, aceptas cumplirlo.
