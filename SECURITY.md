# Política de seguridad

## Versiones soportadas

| Versión | Soporte de seguridad |
|---------|----------------------|
| 0.1.x   | ✅ |
| < 0.1   | ❌ |

## Reportar una vulnerabilidad

**No abras issues públicos para reportar vulnerabilidades.**

1. **Preferido:** usa [GitHub Private Vulnerability Reporting](https://github.com/rellytechgame/Reverse3D-tool/security/advisories/new).
2. **Alternativa:** escribe a **info@voxel.lat** con el asunto `[SECURITY] reverse3d`.

Incluye la versión afectada (`reverse3d --version`), los pasos para reproducirla, el impacto y, si lo tienes, un archivo de prueba mínimo.

| Etapa | Objetivo |
|-------|----------|
| Acuse de recibo | 3 días hábiles |
| Evaluación inicial y severidad (CVSS) | 10 días hábiles |
| Corrección de vulnerabilidades críticas o altas | 30 días |
| Divulgación coordinada | tras publicar la corrección, con crédito al investigador si lo desea |

## Alcance

Están **dentro** del alcance:

- Escapes del sandbox de workspace (path traversal, symlinks, referencias externas en glTF/OBJ).
- Inyección de comandos o argumentos en COLMAP o Blender.
- Ejecución de código al cargar archivos (mallas, imágenes, JSON, modelos de IA).
- Agotamiento de recursos que evada los límites configurados.
- Fugas de secretos del entorno hacia procesos hijos o hacia las respuestas de las herramientas.
- Vulnerabilidades en el servidor MCP o en el despachador de herramientas para agentes.

Están **fuera** del alcance:

- Vulnerabilidades propias de COLMAP, Blender, PyTorch u otras dependencias. Repórtalas a su proyecto; nosotros actualizaremos la versión mínima.
- Ataques que requieren que el operador ejecute comandos CLI arbitrarios con rutas maliciosas. La CLI directa confía en su operador; el sandbox aplica a `agent call` y `agent mcp`.
- Uso de modelos con licencia no comercial fuera de sus términos.

## Suplantación de autoría o del proyecto

Si encuentras un fork, paquete, servidor MCP o extensión que se presenta como el reverse3d oficial o como si su autor fuera otro, repórtalo a **info@voxel.lat** con el asunto `[SUPLANTACIÓN] reverse3d`. En [docs/authenticity.md](docs/authenticity.md) se explica cómo verificar los releases oficiales: tags firmados y atestaciones Sigstore.

## Modelo de seguridad

El diseño completo (activos, amenazas, controles y riesgos residuales) está en [docs/security-model.md](docs/security-model.md).

## Controles automatizados

Cada PR y cada ejecución semanal de CI corren: `bandit` (SAST), `CodeQL`, `pip-audit` (SCA), `gitleaks` (secretos), generación de SBOM (CycloneDX) y la suite de tests. Esta incluye pruebas negativas de path traversal, inyección de argumentos, limpieza del entorno de procesos hijos y validación estricta de los argumentos de las herramientas.
