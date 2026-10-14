# Autenticidad, autoría y protección contra suplantación

**reverse3d fue creado por y es propiedad de [Voxel Enterprise](https://voxel.lat).**
Repositorio oficial: **https://github.com/voxelenterprise/reverse3d**

## Qué se puede y qué no se puede impedir

| Escenario | ¿Se puede impedir? | Cómo |
|-----------|-------------------|------|
| Alguien (persona o agente de IA) cambia autor, licencia o copyright **en el repo oficial** | ✅ **Sí** | Rulesets de GitHub + CODEOWNERS + guardián de autoría en CI y pre-commit |
| Alguien publica una release "oficial" falsa **desde el repo oficial** | ✅ **Sí** | Solo publican tags firmados y verificados, más el entorno `pypi` con tu aprobación |
| Alguien clona el código y cambia el autor **en su copia** | ❌ No técnicamente: es una propiedad de cualquier software open source | La copia **no puede producir tus firmas** ni tus atestaciones, y además viola Apache-2.0 §4 y la [política de marca](../TRADEMARKS.md) |
| Alguien publica otro paquete o servidor MCP llamado "reverse3d" | ⚠️ Se previene en parte | Reservar el nombre en PyPI, registrar la marca y reportar la infracción |

> En resumen: el **repo oficial** queda blindado; las **copias** no se pueden bloquear, pero sí **delatar** (no tienen firma verificable) y **perseguir** legalmente.

---

## Para usuarios y agentes: cómo verificar que es oficial

1. **Origen:** instala solo desde `github.com/voxelenterprise/reverse3d` o desde PyPI `reverse3d`, publicado desde ese repo.
2. **El propio paquete lo indica:**
   ```bash
   reverse3d --version
   # reverse3d 0.2.0 — © Voxel Enterprise (https://voxel.lat) · Apache-2.0
   # Official source: https://github.com/voxelenterprise/reverse3d
   ```
   Una copia puede alterar este texto. Por eso lo que hace prueba son los puntos 3 y 4.
3. **Atestación de procedencia (Sigstore):** cada archivo de una release oficial está firmado por el workflow del repo oficial:
   ```bash
   gh attestation verify reverse3d-0.2.0-py3-none-any.whl --repo voxelenterprise/reverse3d
   gh attestation verify install.sh --repo voxelenterprise/reverse3d
   ```
   Si el archivo no salió del repo oficial, la verificación **falla**.
4. **Firmas de git:** los commits y tags oficiales aparecen como **Verified** en GitHub. Desde un clon:
   ```bash
   git tag -v v0.2.0          # firma del tag de la release
   git log --show-signature   # firmas de los commits
   ```
5. **PyPI:** la página del proyecto muestra *"Trusted Publisher: GitHub voxelenterprise/reverse3d"* y las atestaciones de cada archivo.

### Reportar una suplantación

- Escribe a **info@voxel.lat** con el asunto `[SUPLANTACIÓN] reverse3d` e incluye el enlace.
- GitHub: [reportar contenido o suplantación](https://support.github.com/contact/report-abuse).
- PyPI: [reportar un proyecto malicioso o suplantador](https://pypi.org/security/).
- Directorios de servidores MCP y marketplaces de agentes: usa su canal de reporte de marca.

---

## Para el propietario: lista de blindaje

Estos ajustes se hacen en GitHub y PyPI con la cuenta propietaria. **Ningún archivo del repo puede activarlos por ti**, y son los que impiden cambios de propiedad en el repo oficial.

### 1. Cuenta y organización

- [ ] El repo vive en la **organización** `voxelenterprise`, con la cuenta de la empresa como **Owner**.
- [ ] **2FA obligatorio** para todos los miembros (*Settings → Authentication security*).
- [ ] **Base permissions: Read** para miembros; **escritura solo para quien la necesite**; **Admin solo el propietario**.
- [ ] Solo los Owners pueden transferir, renombrar o borrar el repo y cambiar su visibilidad. No des el rol *Owner* a nadie más.
- [ ] Activa **Vigilant mode** (*Settings → SSH and GPG keys*): los commits que usen tu nombre sin tu firma se marcan como **Unverified**.

### 2. Ruleset de la rama `main` (*Settings → Rules → Rulesets*)

- [ ] Restrict deletions · Block force pushes.
- [ ] Require a pull request before merging · **Require review from Code Owners** · Dismiss stale approvals.
- [ ] **Require signed commits**.
- [ ] Require status checks: `Lint & types` (incluye el **guardián de autoría**), `Tests`, `Security`, `CodeQL`, `Installer`.
- [ ] *Bypass list*: solo tú.

### 3. Ruleset de tags `v*`

- [ ] Restrict creations / updates / deletions → *bypass* **solo para el propietario**. Nadie más puede crear tags de release.
- [ ] Variable de repositorio `RELEASE_MAINTAINERS = tu-usuario` (*Settings → Secrets and variables → Actions → Variables*).

### 4. Entorno `pypi` (*Settings → Environments → pypi*)

- [ ] **Required reviewers: tú.** Nada llega a PyPI sin tu clic.
- [ ] *Deployment branches and tags*: solo tags `v*`.

### 5. PyPI

- [ ] **Reserva el nombre `reverse3d`** publicando la primera versión desde el workflow de release.
- [ ] Configura el **Trusted Publisher**: repo `voxelenterprise/reverse3d`, workflow `release.yml`, entorno `pypi`. No crees tokens de API.
- [ ] 2FA en la cuenta de PyPI.

### 6. Firma de commits y tags (en tu equipo)

Con una clave SSH, la opción más simple:

```bash
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
git config --global commit.gpgsign true
git config --global tag.gpgsign true
```

Sube la misma clave pública a GitHub como **Signing key** (*Settings → SSH and GPG keys → New SSH key → Key type: Signing Key*).

Para publicar una release:

```bash
git tag -s v0.2.0 -m "reverse3d v0.2.0"
git push origin v0.2.0
```

El workflow `release.yml` rechaza tags sin firma verificada.

### 7. Protección legal

- [ ] Registra las marcas **"reverse3d"** y **"Voxel Enterprise"**. En Panamá se hace en la DIGERPI (Dirección General del Registro de la Propiedad Industrial, MICI); evalúa también registros en otros países donde operes.
- [ ] Conserva evidencias de la autoría original: historial de git firmado, fecha del primer commit, releases con atestaciones.

> *Esto no es asesoría legal. Para el registro de marcas y la gestión de infracciones, consulta con un abogado de propiedad intelectual.*

---

## Cómo funciona el guardián de autoría

`scripts/check_authorship.py`:

- Fija las huellas **SHA-256 de `LICENSE` y `NOTICE`**.
- Verifica la autoría en `pyproject.toml`, `reverse3d/_about.py`, `README.md`, `AUTHORS.md` y `TRADEMARKS.md`.
- Comprueba que CODEOWNERS protege todos esos archivos.

Se ejecuta en CI (es un check obligatorio), como hook de pre-commit y en la suite de tests (`tests/test_authorship.py`). Esos tests prueban que cambiar el autor, la URL, la licencia o la regla de CODEOWNERS **se bloquea**.

Para hacer un cambio **legítimo** de esos datos (por ejemplo, el año del copyright), el propietario actualiza el archivo y las huellas del guardián en el mismo PR, que CODEOWNERS obliga a aprobar.
