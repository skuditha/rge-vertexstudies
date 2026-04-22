# RGE Vertex Studies

Project is currently at milestone 1:

- C++ HIPO-to-ROOT extractor
- ROOT output readable by `uproot`
- Python histogram code for `REC::Particle.vz` vs `REC::FTrack.vz`
- Charge, detector-region, and sector breakdowns

No event selection is applied. The extractor keeps all charged `REC::Particle` rows with `pid != 0`.

Detector-region convention:

```text
forward: 2000 < abs(status) < 4000
central: abs(status) > 4000
other:   everything else
```

Build:

```bash
cd cpp
./scripts/build.sh
```

Extract:

```bash
python scripts/01_extract_ntuples.py --runs-config configs/runs.yaml --run 020507
```

Plot:

```bash
python scripts/02_make_vertex_histograms.py --runs-config configs/runs.yaml --run 020507
```