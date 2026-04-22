#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "reader.h"
#include "TFile.h"
#include "TTree.h"

namespace fs = std::filesystem;
float NaN() { return std::numeric_limits<float>::quiet_NaN(); }

std::string trim(const std::string& s) {
    auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::vector<std::string> read_list(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Could not open input list: " + path);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(fin, line)) {
        auto p = line.find('#');
        if (p != std::string::npos) line = line.substr(0, p);
        p = line.find(',');
        if (p != std::string::npos) line = line.substr(0, p);
        line = trim(line);
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

struct Options {
    int run = -1;
    std::string input_list;
    std::string output;
    long long max_events = 0;
    int max_files = 0;
};

void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --run 020507 --input-list files.list --output out.root [--max-events N] [--max-files N]\n";
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto v = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + k);
            return argv[++i];
        };
        if (k == "--run") o.run = std::stoi(v());
        else if (k == "--input-list") o.input_list = v();
        else if (k == "--output") o.output = v();
        else if (k == "--max-events") o.max_events = std::stoll(v());
        else if (k == "--max-files") o.max_files = std::stoi(v());
        else if (k == "--help" || k == "-h") { usage(argv[0]); std::exit(0); }
        else throw std::runtime_error("Unknown option: " + k);
    }
    if (o.run < 0 || o.input_list.empty() || o.output.empty()) throw std::runtime_error("Missing required option.");
    return o;
}

int region_from_status(int status) {
    int a = std::abs(status);
    if (a > 2000 && a < 4000) return 1;
    if (a > 4000) return 2;
    return 0;
}

float mom(float px, float py, float pz) { return std::sqrt(px * px + py * py + pz * pz); }
float theta_deg(float px, float py, float pz) { float p = mom(px, py, pz); return p > 0 ? std::acos(pz / p) * 180.0 / M_PI : NaN(); }
float phi_deg(float px, float py) { return std::atan2(py, px) * 180.0 / M_PI; }

struct FInfo { int ok = 0; float vx = NaN(), vy = NaN(), vz = NaN(), chi2 = NaN(), chi2ndf = NaN(); int sector = 0, ndf = -1; };
struct TInfo { int sector = 0, detector = 0; };

std::map<int, FInfo> ftrack_map(hipo::bank& b) {
    std::map<int, FInfo> m;
    for (int r = 0; r < b.getRows(); ++r) {
        int idx = b.getInt("pindex", r);
        if (idx < 0 || m.count(idx)) continue;
        FInfo x;
        x.ok = 1;
        x.vx = b.getFloat("vx", r);
        x.vy = b.getFloat("vy", r);
        x.vz = b.getFloat("vz", r);
        x.sector = b.getInt("sector", r);
        x.chi2 = b.getFloat("chi2", r);
        x.ndf = b.getInt("NDF", r);
        if (x.ndf > 0) x.chi2ndf = x.chi2 / x.ndf;
        m[idx] = x;
    }
    return m;
}

std::map<int, TInfo> rectrack_map(hipo::bank& b) {
    std::map<int, TInfo> m;
    for (int r = 0; r < b.getRows(); ++r) {
        int idx = b.getInt("pindex", r);
        if (idx < 0 || m.count(idx)) continue;
        TInfo x;
        x.sector = b.getInt("sector", r);
        x.detector = b.getInt("detector", r);
        m[idx] = x;
    }
    return m;
}

int main(int argc, char** argv) {
    try {
        Options o = parse(argc, argv);
        auto files = read_list(o.input_list);
        if (o.max_files > 0 && (int)files.size() > o.max_files) files.resize(o.max_files);
        fs::create_directories(fs::path(o.output).parent_path());

        TFile fout(o.output.c_str(), "RECREATE");
        TTree tracks("tracks", "charged pid!=0 REC::Particle tracks; sector=0 for non-forward tracks");
        TTree run_info("run_info", "extractor counters");

        int run = o.run, file_index = -1, particle_index = -1;
        long long event_index = -1, global_event_id = -1;
        int pid = 0, charge = 0, status = 0, sector = 0, rec_track_detector = 0, detector_region = 0;
        float chi2pid = NaN(), px = NaN(), py = NaN(), pz = NaN(), p = NaN(), theta = NaN(), phi = NaN();
        float vx_particle = NaN(), vy_particle = NaN(), vz_particle = NaN();
        int has_ftrack = 0, ftrack_sector = 0, ftrack_ndf = -1;
        float vx_ftrack = NaN(), vy_ftrack = NaN(), vz_ftrack = NaN(), ftrack_chi2 = NaN(), ftrack_chi2_ndf = NaN();

        tracks.Branch("run", &run, "run/I");
        tracks.Branch("file_index", &file_index, "file_index/I");
        tracks.Branch("event_index", &event_index, "event_index/L");
        tracks.Branch("global_event_id", &global_event_id, "global_event_id/L");
        tracks.Branch("particle_index", &particle_index, "particle_index/I");
        tracks.Branch("pid", &pid, "pid/I");
        tracks.Branch("charge", &charge, "charge/I");
        tracks.Branch("status", &status, "status/I");
        tracks.Branch("sector", &sector, "sector/I");
        tracks.Branch("rec_track_detector", &rec_track_detector, "rec_track_detector/I");
        tracks.Branch("chi2pid", &chi2pid, "chi2pid/F");
        tracks.Branch("px", &px, "px/F"); tracks.Branch("py", &py, "py/F"); tracks.Branch("pz", &pz, "pz/F");
        tracks.Branch("p", &p, "p/F"); tracks.Branch("theta", &theta, "theta/F"); tracks.Branch("phi", &phi, "phi/F");
        tracks.Branch("vx_particle", &vx_particle, "vx_particle/F");
        tracks.Branch("vy_particle", &vy_particle, "vy_particle/F");
        tracks.Branch("vz_particle", &vz_particle, "vz_particle/F");
        tracks.Branch("has_ftrack", &has_ftrack, "has_ftrack/I");
        tracks.Branch("vx_ftrack", &vx_ftrack, "vx_ftrack/F");
        tracks.Branch("vy_ftrack", &vy_ftrack, "vy_ftrack/F");
        tracks.Branch("vz_ftrack", &vz_ftrack, "vz_ftrack/F");
        tracks.Branch("ftrack_sector", &ftrack_sector, "ftrack_sector/I");
        tracks.Branch("ftrack_chi2", &ftrack_chi2, "ftrack_chi2/F");
        tracks.Branch("ftrack_ndf", &ftrack_ndf, "ftrack_ndf/I");
        tracks.Branch("ftrack_chi2_ndf", &ftrack_chi2_ndf, "ftrack_chi2_ndf/F");
        tracks.Branch("detector_region", &detector_region, "detector_region/I");

        long long n_files_requested = files.size(), n_files_opened = 0, n_events_seen = 0, n_tracks_seen = 0, n_tracks_written = 0, n_tracks_with_ftrack = 0;
        run_info.Branch("run", &run, "run/I");
        run_info.Branch("n_files_requested", &n_files_requested, "n_files_requested/L");
        run_info.Branch("n_files_opened", &n_files_opened, "n_files_opened/L");
        run_info.Branch("n_events_seen", &n_events_seen, "n_events_seen/L");
        run_info.Branch("n_tracks_seen", &n_tracks_seen, "n_tracks_seen/L");
        run_info.Branch("n_tracks_written", &n_tracks_written, "n_tracks_written/L");
        run_info.Branch("n_tracks_with_ftrack", &n_tracks_with_ftrack, "n_tracks_with_ftrack/L");

        for (file_index = 0; file_index < (int)files.size(); ++file_index) {
            if (!fs::exists(files[file_index])) { std::cerr << "Missing file, skipping: " << files[file_index] << "\n"; continue; }
            std::cout << "Reading " << file_index + 1 << "/" << files.size() << ": " << files[file_index] << "\n";
            hipo::reader reader; reader.open(files[file_index].c_str());
            hipo::dictionary dict; reader.readDictionary(dict);
            hipo::bank particles(dict.getSchema("REC::Particle"));
            hipo::bank ftracks(dict.getSchema("REC::FTrack"));
            hipo::bank rectracks(dict.getSchema("REC::Track"));
            hipo::event event;
            ++n_files_opened; event_index = -1;
            while (reader.next()) {
                reader.read(event);
                event.getStructure(particles); event.getStructure(ftracks); event.getStructure(rectracks);
                ++event_index; ++global_event_id; ++n_events_seen;
                auto fmap = ftrack_map(ftracks);
                auto tmap = rectrack_map(rectracks);
                for (int row = 0; row < particles.getRows(); ++row) {
                    ++n_tracks_seen;
                    pid = particles.getInt("pid", row);
                    charge = particles.getInt("charge", row);
                    if (pid == 0 || charge == 0) continue;
                    particle_index = row;
                    px = particles.getFloat("px", row); py = particles.getFloat("py", row); pz = particles.getFloat("pz", row);
                    p = mom(px, py, pz); theta = theta_deg(px, py, pz); phi = phi_deg(px, py);
                    vx_particle = particles.getFloat("vx", row); vy_particle = particles.getFloat("vy", row); vz_particle = particles.getFloat("vz", row);
                    status = particles.getInt("status", row); chi2pid = particles.getFloat("chi2pid", row);
                    detector_region = region_from_status(status);
                    sector = 0; rec_track_detector = 0;
                    if (tmap.count(row)) { sector = tmap[row].sector; rec_track_detector = tmap[row].detector; }
                    has_ftrack = 0; vx_ftrack = NaN(); vy_ftrack = NaN(); vz_ftrack = NaN(); ftrack_sector = 0; ftrack_chi2 = NaN(); ftrack_ndf = -1; ftrack_chi2_ndf = NaN();
                    if (fmap.count(row)) {
                        auto f = fmap[row]; has_ftrack = 1; vx_ftrack = f.vx; vy_ftrack = f.vy; vz_ftrack = f.vz;
                        ftrack_sector = f.sector; ftrack_chi2 = f.chi2; ftrack_ndf = f.ndf; ftrack_chi2_ndf = f.chi2ndf;
                        if (sector == 0) sector = ftrack_sector;
                        ++n_tracks_with_ftrack;
                    }

                    // Only the forward detector has a meaningful six-sector split
                    // for this analysis. Treat central and other detector-region
                    // tracks as a single sectorless category.
                    if (detector_region != 1) {
                        sector = 0;
                    }

                    tracks.Fill(); ++n_tracks_written;
                }
                if (o.max_events > 0 && n_events_seen >= o.max_events) break;
            }
            if (o.max_events > 0 && n_events_seen >= o.max_events) break;
        }
        run_info.Fill();
        fout.cd(); tracks.Write(); run_info.Write(); fout.Close();
        std::cout << "Done. Wrote " << n_tracks_written << " tracks to " << o.output << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n"; usage(argv[0]); return 1;
    }
}