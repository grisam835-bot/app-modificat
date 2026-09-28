import streamlit as st
import numpy as np
from scipy.optimize import minimize, root_scalar
from scipy.stats import poisson, nbinom, erlang, wasserstein_distance
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pure Market Engine 9x9 (Optimized + Copula + Toothprints)", layout="wide")

class PureMarketEngine9x9:
    def __init__(self, max_goals=8):
        self.max_goals = max_goals
        self.weights = self._build_weight_matrix()

    def _build_weight_matrix(self):
        W = np.full((self.max_goals + 1, self.max_goals + 1), 0.02)
        anchor_scores = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (2,1), (1,2), (3,0), (0,3)]
        for h, a in anchor_scores:
            if h <= self.max_goals and a <= self.max_goals:
                W[h, a] = 1.0
        secondary_scores = [(3,1), (1,3), (2,2), (3,2), (2,3), (4,0), (0,4), (4,1), (1,4), (4,2), (2,4)]
        for h, a in secondary_scores:
            if h <= self.max_goals and a <= self.max_goals:
                W[h, a] = 0.45
        return W

    def _power_unmargin_binary(self, odd_1, odd_2):
        if odd_1 <= 1.0 or odd_2 <= 1.0:
            return None, None
        p1_raw, p2_raw = 1.0 / odd_1, 1.0 / odd_2
        try:
            res = root_scalar(lambda k: (p1_raw ** k + p2_raw ** k) - 1.0, bracket=[0.001, 5.0], method='brentq')
            return (p1_raw ** res.root), (p2_raw ** res.root)
        except:
            clean = self._shin_unmargin({'1': p1_raw, '2': p2_raw})
            return clean['1'], clean['2']

    def _shin_unmargin(self, raw_probs):
        n = len(raw_probs)
        if n == 0: return {}
        sum_pi = sum(raw_probs.values())
        if sum_pi <= 1.0: return {k: v / sum_pi for k, v in raw_probs.items()}
        z = (sum_pi - 1.0) / max(1, n - 1)
        clean_probs = {}
        for k, pi in raw_probs.items():
            num = np.sqrt(z**2 + 4 * (1 - z) * (pi**2 / sum_pi)) - z
            den = 2 * (1 - z)
            clean_probs[k] = max(1e-12, num / max(den, 1e-12))
        total_clean = sum(clean_probs.values())
        return {k: v / total_clean for k, v in clean_probs.items()}

    def _shin_unmargin_asymmetric(self, raw_probs, z_mri_x=0.0):
        if z_mri_x <= 1.8:
            return self._shin_unmargin(raw_probs)
        
        n = len(raw_probs)
        if n == 0: return {}
        sum_pi = sum(raw_probs.values())
        if sum_pi <= 1.0: return {k: v / sum_pi for k, v in raw_probs.items()}
        
        z_base = (sum_pi - 1.0) / max(1, n - 1)
        clean_probs = {}
        asym_boost = 1.0 + 0.12 * min(z_mri_x - 1.8, 2.5)
        
        for k, pi in raw_probs.items():
            z_eff = z_base * (asym_boost if (isinstance(k, tuple) and k[0] == k[1]) else 1.0)
            num = np.sqrt(z_eff**2 + 4 * (1 - z_eff) * (pi**2 / sum_pi)) - z_eff
            den = 2 * (1 - z_eff)
            clean_probs[k] = max(1e-12, num / max(den, 1e-12))
            
        total_clean = sum(clean_probs.values())
        return {k: v / total_clean for k, v in clean_probs.items()}

    def calculate_x_stress_z_score(self, kl_div, x_gap, mri_index):
        mri_x_component = mri_index * (x_gap / max(x_gap + kl_div + 0.1, 1e-4))
        z_mri_x = (mri_x_component - 15.0) / 12.0
        return float(max(0.0, z_mri_x))

    def _apply_shannon_bayes_noise_filter(self, raw_probs):
        if not raw_probs: return {}
        sum_pi = sum(raw_probs.values())
        payout = 1.0 / sum_pi if sum_pi > 0 else 1.0
        if payout < 0.92:
            filtered_probs = {}
            margin_penalty = (0.92 - payout) * 2.0
            for k, p in raw_probs.items():
                sigmoid_weight = 1.0 / (1.0 + np.exp(-15.0 * (p - 0.03)))
                adj_p = p * (1.0 - margin_penalty * (1.0 - sigmoid_weight))
                filtered_probs[k] = max(1e-12, adj_p)
            return filtered_probs
        return raw_probs

    def _get_shin_alpha(self, raw_probs):
        n = len(raw_probs)
        if n <= 1: return 0.0
        sum_pi = sum(raw_probs.values())
        if sum_pi <= 1.0: return 0.0
        return float((sum_pi - 1.0) / max(1, n - 1))

    def _apply_copula_density(self, u, v, copula_type="Frank", theta=1.5):
        u = np.clip(u, 1e-6, 1.0 - 1e-6)
        v = np.clip(v, 1e-6, 1.0 - 1e-6)
        
        if copula_type == "Frank":
            if abs(theta) < 1e-4:
                return np.ones_like(u) if isinstance(u, np.ndarray) else 1.0
            num = -theta * (np.exp(-theta) - 1.0) * np.exp(-theta * (u + v))
            den = ((np.exp(-theta * u) - 1.0) * (np.exp(-theta * v) - 1.0) + (np.exp(-theta) - 1.0)) ** 2
            return np.maximum(1e-6, num / np.maximum(den, 1e-12))
            
        elif copula_type == "Gumbel":
            if theta <= 1.0:
                return np.ones_like(u) if isinstance(u, np.ndarray) else 1.0
            x = -np.log(u)
            y = -np.log(v)
            A = (x**theta + y**theta)**(1.0 / theta)
            C = np.exp(-A)
            density = (C / (u * v)) * ((x * y)**(theta - 1.0) / (x**theta + y**theta)**(2.0 - 1.0/theta)) * (A + theta - 1.0)
            return np.maximum(1e-6, density)

        elif copula_type == "Clayton":
            if theta <= 0.0:
                return np.ones_like(u) if isinstance(u, np.ndarray) else 1.0
            density = (1.0 + theta) * ((u * v) ** (-1.0 - theta)) * ((u ** (-theta) + v ** (-theta) - 1.0) ** (-2.0 - 1.0 / theta))
            return np.maximum(1e-6, density)
            
        return np.ones_like(u) if isinstance(u, np.ndarray) else 1.0

    def _get_marginal_pmf_cdf(self, mu, k_val, phi=0.0):
        if phi <= 1e-4:
            pmf = poisson.pmf(k_val, mu)
            cdf = poisson.cdf(k_val, mu)
        else:
            r = 1.0 / phi
            p_param = r / (r + mu)
            pmf = nbinom.pmf(k_val, r, p_param)
            cdf = nbinom.cdf(k_val, r, p_param)
        return pmf, cdf

    def _generate_matrix(self, lambda_h, mu_a, rho=0.0, pi_zero=0.0, copula_type="Fără", copula_theta=1.5, phi_dispersion=0.0):
        h_arr = np.arange(self.max_goals + 1)
        a_arr = np.arange(self.max_goals + 1)
        
        p_h, cdf_h = self._get_marginal_pmf_cdf(lambda_h, h_arr, phi_dispersion)
        p_a, cdf_a = self._get_marginal_pmf_cdf(mu_a, a_arr, phi_dispersion)
        
        P_H, P_A = np.meshgrid(p_h, p_a, indexing='ij')
        CDF_H, CDF_A = np.meshgrid(cdf_h, cdf_a, indexing='ij')
        
        xg_ratio = lambda_h / max(mu_a, 1e-5)
        if xg_ratio > 2.5:
            asym_factor = 1.0 + 0.15 * np.tanh(xg_ratio - 2.5)
        elif xg_ratio < 0.4:
            asym_factor = 1.0 + 0.15 * np.tanh((1.0 / xg_ratio) - 2.5)
        else:
            asym_factor = 1.0
        eff_rho = rho * asym_factor

        adj = np.ones((self.max_goals + 1, self.max_goals + 1))
        if self.max_goals >= 1:
            adj[0, 0] = 1.0 - (lambda_h * mu_a * eff_rho)
            adj[1, 0] = 1.0 + (mu_a * eff_rho)
            adj[0, 1] = 1.0 + (lambda_h * eff_rho)
            adj[1, 1] = 1.0 - eff_rho

        base_matrix = P_H * P_A * adj
        
        if copula_type != "Fără":
            u_mid = np.clip(CDF_H - 0.5 * P_H, 1e-6, 1.0 - 1e-6)
            v_mid = np.clip(CDF_A - 0.5 * P_A, 1e-6, 1.0 - 1e-6)
            c_density = self._apply_copula_density(u_mid, v_mid, copula_type, copula_theta)
            base_matrix *= c_density

        diag_mask = np.eye(self.max_goals + 1, dtype=bool)
        diag_boost = 1.0 + (pi_zero * np.exp(-0.5 * h_arr))
        
        matrix = np.where(diag_mask, base_matrix * diag_boost, (1.0 - pi_zero * 0.2) * base_matrix)
        matrix = np.maximum(1e-12, matrix)
                
        total_p = np.sum(matrix)
        return matrix / total_p if total_p > 0 else matrix

    def _huber_loss(self, y_true, y_pred, delta=0.001):
        error = y_pred - y_true
        return np.where(np.abs(error) <= delta, 0.5 * (error ** 2), delta * (np.abs(error) - 0.5 * delta))

    # =========================================================================
    # REVERSE ENGINEERING SERVER CENTRAL & RESIDUAL ERROR
    # =========================================================================

    def _calculate_model_residual_error(self, matrix, cs_odds, main_ah, main_ou):
        """Calculează SSE (Sum of Squared Errors) între matrice și cotele reale."""
        sse = 0.0
        valid_cs = {k: 1.0/v for k, v in cs_odds.items() if 1.0 < v <= 100.0}
        clean_cs = self._shin_unmargin(valid_cs)
        
        for (h, a), p_real in clean_cs.items():
            if h <= self.max_goals and a <= self.max_goals:
                sse += (matrix[h, a] - p_real) ** 2
                
        if main_ah and main_ah.get('home_odd', 0) > 1.0 and main_ah.get('away_odd', 0) > 1.0:
            p_ah_clean, _ = self._power_unmargin_binary(main_ah['home_odd'], main_ah['away_odd'])
            if p_ah_clean:
                p_ah_calc = self.calculate_ah_probability(matrix, main_ah['home_line'], is_home=True)
                sse += (p_ah_calc - p_ah_clean) ** 2

        if main_ou and main_ou.get('over_odd', 0) > 1.0 and main_ou.get('under_odd', 0) > 1.0:
            p_ou_clean, _ = self._power_unmargin_binary(main_ou['over_odd'], main_ou['under_odd'])
            if p_ou_clean:
                p_ou_calc = self.calculate_ou_probability(matrix, main_ou['line'], is_over=True)
                sse += (p_ou_calc - p_ou_clean) ** 2

        return float(sse)

    def reverse_engineer_bookmaker_state(self, cs_odds, main_ah, main_ou, main_x):
        """
        Reverse Engineering pe Serverul Central al Casei de Pariuri.
        Află exact ce Vector de Stare θ și ce Copula rulează casa.
        """
        copula_candidates = ["Frank", "Gumbel", "Clayton"]
        best_fit = None
        min_aic = float('inf')
        
        n_data_points = len(cs_odds) + 3
        
        for c_type in copula_candidates:
            l, m, r, pi, _, _, _, _, _, _, mat, theta_fit, phi_fit = self.extract_pure_xg(
                cs_odds, main_ah, main_ou, main_x, 
                mode="Anchored (5.0 Weight)", 
                copula_type=c_type, 
                auto_fit_copula=True, 
                use_nbinom=True
            )
            
            sse = self._calculate_model_residual_error(mat, cs_odds, main_ah, main_ou)
            k_params = 6 
            aic = n_data_points * np.log(max(sse / n_data_points, 1e-12)) + 2 * k_params
            
            if aic < min_aic:
                min_aic = aic
                best_fit = {
                    'copula_type': c_type,
                    'lambda_home': round(l, 3),
                    'mu_away': round(m, 3),
                    'dixon_coles_rho': round(r, 4),
                    'copula_theta': round(theta_fit, 3),
                    'nbinom_phi': round(phi_fit, 4),
                    'aic_score': round(aic, 2),
                    'fit_quality': "EXCELENT" if sse < 0.005 else "MEDIU / ZGOMOT"
                }
                
        return best_fit

    # =========================================================================
    # METODELE ANTERIOARE (MODULELE 1-7)
    # =========================================================================

    def calculate_cross_market_stress_kl(self, mat_decoupled, mat_close, main_ou_close):
        p_decoupled = np.clip(mat_decoupled, 1e-12, 1.0)
        p_close = np.clip(mat_close, 1e-12, 1.0)
        
        p_home_dec = np.sum(np.tril(p_decoupled, -1))
        p_draw_dec = np.trace(p_decoupled)
        p_away_dec = np.sum(np.triu(p_decoupled, 1))
        p_1x2_dec = np.array([p_home_dec, p_draw_dec, p_away_dec])
        p_1x2_dec /= np.sum(p_1x2_dec)

        p_home_cls = np.sum(np.tril(p_close, -1))
        p_draw_cls = np.trace(p_close)
        p_away_cls = np.sum(np.triu(p_close, 1))
        p_1x2_cls = np.array([p_home_cls, p_draw_cls, p_away_cls])
        p_1x2_cls /= np.sum(p_1x2_cls)

        kl_1x2 = np.sum(p_1x2_cls * np.log(p_1x2_cls / p_1x2_dec))

        p_over_dec = self.calculate_ou_probability(p_decoupled, main_ou_close['line'], is_over=True)
        p_under_dec = 1.0 - p_over_dec
        p_ou_dec = np.array([p_over_dec, p_under_dec])

        p_over_cls = self.calculate_ou_probability(p_close, main_ou_close['line'], is_over=True)
        p_under_cls = 1.0 - p_over_cls
        p_ou_cls = np.array([p_over_cls, p_under_cls])

        kl_ou = np.sum(p_ou_cls * np.log(p_ou_cls / p_ou_dec))

        cmsi_kl = abs(kl_1x2 - kl_ou) * 100.0
        return float(np.clip(cmsi_kl, 0.0, 1.0))

    def calculate_asymmetric_pressure_field(self, mat_close):
        grad_y, grad_x = np.gradient(mat_close)
        grad_2_y, grad_2_x = np.gradient(grad_y)[0], np.gradient(grad_x)[1]
        
        diag_curvature = np.diag(grad_2_y + grad_2_x)
        home_side_pressure = np.sum(grad_y[np.tril_indices(self.max_goals + 1, -1)])
        away_side_pressure = np.sum(grad_x[np.triu_indices(self.max_goals + 1, 1)])
        
        apf_score = (home_side_pressure - away_side_pressure) / (abs(home_side_pressure + away_side_pressure) + 1e-6)
        diag_asymmetry = float(np.std(diag_curvature))
        return float(apf_score), diag_asymmetry, diag_curvature

    def calculate_cell_level_vig_squeeze(self, mat_decoupled, mat_close_raw):
        p_dec = np.maximum(1e-6, mat_decoupled)
        p_cls = np.maximum(1e-6, mat_close_raw)
        
        vig_squeeze_matrix = p_cls / p_dec
        max_squeeze = float(np.max(vig_squeeze_matrix))
        squeezed_cells_idx = np.argwhere(vig_squeeze_matrix > 1.25)
        squeezed_cells = [(int(c[0]), int(c[1]), float(vig_squeeze_matrix[c[0], c[1]])) for c in squeezed_cells_idx]
        return vig_squeeze_matrix, max_squeeze, squeezed_cells

    def calculate_time_decay_drift(self, mat_open, mat_close, total_xg_close):
        delta_p = mat_close - mat_open
        abs_drift = np.sum(np.abs(delta_p))
        decay_acceleration = float(abs_drift * np.exp(-0.15 * total_xg_close))
        drift_direction = "Inflow (Concentrare)" if np.sum(delta_p) > 0 else "Outflow (Dispersie)"
        return float(abs_drift), decay_acceleration, drift_direction

    def calculate_market_entropy_vulnerability_index(self, mat_close, ent_close, gini_close):
        draw_mass = np.trace(mat_close)
        mevi_score = (ent_close / (gini_close + 1e-5)) * (1.0 - draw_mass)
        vulnerability_level = "CRITICĂ" if mevi_score > 15.0 else ("MODERATĂ" if mevi_score > 8.0 else "SCĂZUTĂ")
        return float(mevi_score), vulnerability_level

    def calculate_modulo_7_integrated_metrics(self, mat_open, mat_close, ent_close, gini_close, total_xg):
        abs_drift, decay_acc, drift_dir = self.calculate_time_decay_drift(mat_open, mat_close, total_xg)
        mevi_score, vuln_level = self.calculate_market_entropy_vulnerability_index(mat_close, ent_close, gini_close)
        return {
            'abs_drift': round(abs_drift, 4),
            'decay_acc': round(decay_acc, 4),
            'drift_dir': drift_dir,
            'mevi_score': round(mevi_score, 3),
            'vuln_level': vuln_level
        }

    def calculate_vector_field_flow(self, mat_open, mat_close):
        delta_p = mat_close - mat_open
        v_y, u_x = np.gradient(delta_p)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(delta_p, cmap="coolwarm", origin="upper", vmin=-0.03, vmax=0.03)
        x, y = np.meshgrid(np.arange(self.max_goals + 1), np.arange(self.max_goals + 1))
        ax.quiver(x, y, u_x, v_y, color="black", angles="xy", scale_units="xy", scale=0.5, pivot="middle")
        ax.set_title("Vector Field Flow (ΔP & Gradient Stream)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Goluri Oaspeți")
        ax.set_ylabel("Goluri Gazde")
        ax.set_xticks(range(self.max_goals + 1))
        ax.set_yticks(range(self.max_goals + 1))
        fig.colorbar(cax, ax=ax, label="Shift Probabilitate (ΔP)")
        plt.tight_layout()
        plt.close(fig)
        return u_x, v_y, fig

    def calculate_surface_laplacian(self, mat_p):
        kernel = np.array([[0,  1, 0], [1, -4, 1], [0,  1, 0]])
        laplacian_map = ndimage.convolve(mat_p, kernel, mode='constant', cval=0.0)
        return laplacian_map, float(np.max(np.abs(laplacian_map))), float(np.std(laplacian_map))

    def calculate_topological_bimodal_index(self, mat_p, threshold_relative=0.25):
        local_max = ndimage.maximum_filter(mat_p, size=3) == mat_p
        significant_peaks = local_max & (mat_p >= (np.max(mat_p) * threshold_relative))
        coords = np.argwhere(significant_peaks)
        return len(coords) >= 2, len(coords), [(int(c[0]), int(c[1]), float(mat_p[c[0], c[1]])) for c in coords]

    def calculate_bivariate_moments(self, mat_p):
        grid = np.arange(self.max_goals + 1)
        x_grid, y_grid = np.meshgrid(grid, grid)
        lambda_h = np.sum(mat_p * y_grid)
        mu_a = np.sum(mat_p * x_grid)
        sigma_h = max(np.sqrt(np.sum(mat_p * ((y_grid - lambda_h) ** 2))), 1e-6)
        sigma_a = max(np.sqrt(np.sum(mat_p * ((x_grid - mu_a) ** 2))), 1e-6)
        coskew_ha = np.sum(mat_p * (y_grid - lambda_h) * ((x_grid - mu_a) ** 2)) / (sigma_h * (sigma_a ** 2))
        
        if coskew_ha > 0.10: reg, desc = "Meci Răzbunător (Tit-for-Tat)", "Un gol primit determină o replică ofensivă."
        elif coskew_ha < -0.10: reg, desc = "Meci de Blocaj Defensiv", "Un gol marcat închide jocul."
        else: reg, desc = "Meci Simetric / Echilibrat", "Evoluția scorului urmează dinamica Poisson standard."
        return float(coskew_ha), reg, desc

    def calculate_first_passage_time(self, lambda_h, mu_a):
        total_rate = lambda_h + mu_a
        if total_rate <= 0: return 90.0, 0.0, 0.0
        return float((1.0 / total_rate) * 90.0), float((1.0 - np.exp(-total_rate * (15.0 / 90.0))) * 100), float((1.0 - np.exp(-total_rate * (30.0 / 90.0))) * 100)

    def calculate_gini_index(self, matrix):
        flat = np.sort(matrix.flatten())
        n = len(flat)
        return float((2 * np.sum(np.arange(1, n + 1) * flat)) / (n * np.sum(flat)) - (n + 1) / n)

    def calculate_top3_density(self, matrix):
        return float(np.sum(np.sort(matrix.flatten())[::-1][:3]) * 100.0)

    def calculate_modal_skewness(self, lambda_h, mu_a):
        return float((1.0 / np.sqrt(max(lambda_h, 1e-5))) - (1.0 / np.sqrt(max(mu_a, 1e-5))))

    def calculate_mvi(self, lambda_h, mu_a):
        return float(abs(lambda_h - mu_a) / max(lambda_h + mu_a, 1e-5))

    def calculate_kl_divergence(self, mat_open, mat_close):
        p, q = np.clip(mat_open.flatten(), 1e-12, 1.0), np.clip(mat_close.flatten(), 1e-12, 1.0)
        return float(np.sum(p * np.log(p / q)))

    def calculate_jsd(self, mat_open, mat_close):
        p, q = np.clip(mat_open.flatten(), 1e-12, 1.0), np.clip(mat_close.flatten(), 1e-12, 1.0)
        m = 0.5 * (p + q)
        return float(0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m))))

    def calculate_market_refractive_index(self, jsd_div, ah_line_open, ah_line_close):
        return float(jsd_div / (abs(ah_line_close - ah_line_open) + 1e-4))

    def calculate_erlang_adjusted_srp(self, jsd_div, ah_line_open, ah_line_close, total_xg):
        delta_line = abs(ah_line_close - ah_line_open)
        return float((jsd_div / (delta_line + 1e-4)) * (1.0 + erlang.pdf(delta_line + 0.1, 2, scale=max(0.1, total_xg / 2.0))))

    def calculate_bivariate_skewness_tensor(self, matrix):
        h_grid, a_grid = np.indices(matrix.shape)
        mean_h, mean_a = np.sum(h_grid * matrix), np.sum(a_grid * matrix)
        var_h, var_a = np.sum(((h_grid - mean_h) ** 2) * matrix), np.sum(((a_grid - mean_a) ** 2) * matrix)
        return float((np.sum(((h_grid - mean_h) ** 3) * matrix) / (var_h ** 1.5 + 1e-6)) - (np.sum(((a_grid - mean_a) ** 3) * matrix) / (var_a ** 1.5 + 1e-6)))

    def calculate_analytical_tail_dependence(self, matrix, min_goals=3):
        h_grid, a_grid = np.indices(matrix.shape)
        mask = (h_grid + a_grid) >= min_goals
        if not np.any(mask): return 0.0
        w_norm = matrix[mask] / np.sum(matrix[mask])
        h_vals, a_vals = h_grid[mask], a_grid[mask]
        cov = np.sum((h_vals - np.sum(h_vals * w_norm)) * (a_vals - np.sum(a_vals * w_norm)) * w_norm)
        denom = np.sqrt(np.sum(((h_vals - np.sum(h_vals * w_norm))**2) * w_norm) * np.sum(((a_vals - np.sum(a_vals * w_norm))**2) * w_norm))
        return float(cov / denom) if denom > 1e-6 else 0.0

    def calculate_shin_alpha_variance(self, cs_odds_dict, main_ah_input, main_ou_input, main_x_odd):
        alphas = []
        valid_cs = {k: 1.0 / v for k, v in cs_odds_dict.items() if 1.0 < v <= 100.0}
        if valid_cs: alphas.append(self._get_shin_alpha(valid_cs))
        if main_ah_input and main_ah_input.get('home_odd', 0) > 1.0 and main_ah_input.get('away_odd', 0) > 1.0:
            alphas.append(self._get_shin_alpha({'h': 1.0/main_ah_input['home_odd'], 'a': 1.0/main_ah_input['away_odd']}))
        if main_ou_input and main_ou_input.get('over_odd', 0) > 1.0 and main_ou_input.get('under_odd', 0) > 1.0:
            alphas.append(self._get_shin_alpha({'o': 1.0/main_ou_input['over_odd'], 'u': 1.0/main_ou_input['under_odd']}))
        return float(np.var(alphas)) if len(alphas) >= 2 else 0.0

    def calculate_cross_market_stress_index(self, kl_div, shin_var, x_gap):
        return float(np.clip((0.40 * min(kl_div * 10.0, 1.0)) + (0.35 * min(shin_var * 1000.0, 1.0)) + (0.25 * min(x_gap / 3.0, 1.0)), 0.0, 1.0))

    def calculate_ah_probability(self, matrix, ah_line, is_home=True):
        if abs((abs(ah_line) % 0.5) - 0.25) < 1e-4:
            p_w1, p_p1, _ = self._single_line_ah_outcomes(matrix, ah_line - 0.25, is_home)
            p_w2, p_p2, _ = self._single_line_ah_outcomes(matrix, ah_line + 0.25, is_home)
            return float(np.clip(0.5 * (p_w1 + 0.5 * p_p1 + p_w2 + 0.5 * p_p2), 1e-5, 1.0))
        p_w, p_p, _ = self._single_line_ah_outcomes(matrix, ah_line, is_home)
        return float(p_w / (1.0 - p_p)) if (1.0 - p_p) > 0 else 0.5

    def _single_line_ah_outcomes(self, matrix, ah_line, is_home=True):
        h_grid, a_grid = np.indices(matrix.shape)
        diff = ((h_grid - a_grid) if is_home else (a_grid - h_grid)) + ah_line
        return np.sum(matrix[diff > 1e-5]), np.sum(matrix[np.abs(diff) <= 1e-5]), np.sum(matrix[diff < -1e-5])

    def calculate_ou_probability(self, matrix, ou_line, is_over=True):
        if abs((abs(ou_line) % 0.5) - 0.25) < 1e-4:
            p_w1, p_p1, _ = self._single_line_ou_outcomes(matrix, ou_line - 0.25, is_over)
            p_w2, p_p2, _ = self._single_line_ou_outcomes(matrix, ou_line + 0.25, is_over)
            return float(np.clip(0.5 * (p_w1 + 0.5 * p_p1 + p_w2 + 0.5 * p_p2), 1e-5, 1.0))
        p_w, p_p, _ = self._single_line_ou_outcomes(matrix, ou_line, is_over)
        return float(p_w / (1.0 - p_p)) if (1.0 - p_p) > 0 else 0.5

    def _single_line_ou_outcomes(self, matrix, ou_line, is_over=True):
        h_grid, a_grid = np.indices(matrix.shape)
        diff = ((h_grid + a_grid) - ou_line) if is_over else (ou_line - (h_grid + a_grid))
        return np.sum(matrix[diff > 1e-5]), np.sum(matrix[np.abs(diff) <= 1e-5]), np.sum(matrix[diff < -1e-5])

    def calculate_draw_probability(self, matrix):
        return float(np.trace(matrix))

    def analyze_mass_shifts(self, cs_open_dict, cs_close_dict, ou_line=2.5, z_mri_x=0.0):
        open_clean = self._shin_unmargin_asymmetric(self._apply_shannon_bayes_noise_filter({k: 1.0/v for k, v in cs_open_dict.items() if 1.0 < v <= 100.0}), z_mri_x)
        close_clean = self._shin_unmargin_asymmetric(self._apply_shannon_bayes_noise_filter({k: 1.0/v for k, v in cs_close_dict.items() if 1.0 < v <= 100.0}), z_mri_x)
        shift_report, total_home, total_away, total_under = {}, 0.0, 0.0, 0.0

        for score, p_open in open_clean.items():
            if score in close_clean:
                delta_p = close_clean[score] - p_open
                shift_report[score] = delta_p
                if score[0] > score[1]: total_home += delta_p
                elif score[1] > score[0]: total_away += delta_p
                if (score[0] + score[1]) < ou_line: total_under += delta_p

        return {
            'home_shift_pct': round(total_home * 100, 2),
            'away_shift_pct': round(total_away * 100, 2),
            'under_shift_pct': round(total_under * 100, 2),
            'top_inflows': [(f"{s[0]}-{s[1]}", round(d * 100, 2)) for s, d in sorted(shift_report.items(), key=lambda x: x[1], reverse=True)[:3]],
            'top_outflows': [(f"{s[0]}-{s[1]}", round(d * 100, 2)) for s, d in sorted(shift_report.items(), key=lambda x: x[1])[:3]]
        }

    def extract_pure_xg(self, cs_odds_dict, main_ah_input=None, main_ou_input=None, main_x_odd=None, mode="Decoupled", copula_type="Fără", copula_theta=1.5, auto_fit_copula=False, use_nbinom=False, z_mri_x=0.0):
        clean_probs = self._shin_unmargin_asymmetric(self._apply_shannon_bayes_noise_filter({k: 1.0 / v for k, v in cs_odds_dict.items() if 1.0 < v <= 100.0}), z_mri_x)

        target_p_ah, target_p_ou, target_p_x = None, None, None
        if main_ah_input and main_ah_input.get('home_odd', 0) > 1.0 and main_ah_input.get('away_odd', 0) > 1.0:
            p_h_clean, _ = self._power_unmargin_binary(main_ah_input['home_odd'], main_ah_input['away_odd'])
            target_p_ah = p_h_clean
        if main_ou_input and main_ou_input.get('over_odd', 0) > 1.0 and main_ou_input.get('under_odd', 0) > 1.0:
            p_over_clean, _ = self._power_unmargin_binary(main_ou_input['over_odd'], main_ou_input['under_odd'])
            target_p_ou = p_over_clean
        if main_x_odd and main_x_odd > 1.0:
            target_p_x = (1.0 / main_x_odd) / 1.05

        def loss_function(params):
            lambda_h, mu_a, rho, pi_zero = params[:4]
            current_theta = params[4] if auto_fit_copula and len(params) > 4 else copula_theta
            current_phi = params[5] if use_nbinom and len(params) > 5 else 0.0

            theo_matrix = self._generate_matrix(lambda_h, mu_a, rho, pi_zero, copula_type, current_theta, current_phi)
            sum_theo_subset = sum(theo_matrix[h, a] for (h, a) in clean_probs.keys() if h <= self.max_goals and a <= self.max_goals)
            if sum_theo_subset <= 0: return 1e6
                
            total_loss = 0.0
            for (h, a), p_clean in clean_probs.items():
                if h <= self.max_goals and a <= self.max_goals:
                    total_loss += self.weights[h, a] * self._huber_loss(p_clean, theo_matrix[h, a] / sum_theo_subset, delta=0.001)

            if mode == "Anchored (5.0 Weight)":
                if target_p_ou and main_ou_input:
                    total_loss += 5.0 * ((self.calculate_ou_probability(theo_matrix, main_ou_input['line'], is_over=True) - target_p_ou) ** 2)
                if target_p_ah and main_ah_input:
                    total_loss += 5.0 * ((self.calculate_ah_probability(theo_matrix, main_ah_input['home_line'], is_home=True) - target_p_ah) ** 2)

            if target_p_x:
                total_loss += 2.5 * ((self.calculate_draw_probability(theo_matrix) - target_p_x) ** 2)
                    
            return total_loss

        init_guess = [1.40, 1.10, 0.0, 0.01]
        bounds = [(0.1, 4.5), (0.1, 4.5), (-0.25, 0.25), (0.0, 0.25)]

        if auto_fit_copula and copula_type != "Fără":
            init_guess.append(copula_theta)
            bounds.append((1.01 if copula_type == "Gumbel" else (0.01 if copula_type == "Clayton" else 0.1), 5.0 if copula_type != "Frank" else 10.0))

        if use_nbinom:
            init_guess.append(0.01)
            bounds.append((0.0, 0.5))
        
        res = minimize(loss_function, init_guess, bounds=bounds, method='L-BFGS-B')
        if not res.success or res.x[0] >= 4.3 or res.x[1] >= 4.3:
            res = minimize(loss_function, init_guess, bounds=bounds, method='Nelder-Mead')

        lambda_pure, mu_pure, rho_pure, pi_zero_pure = res.x[:4]
        idx = 4
        fitted_theta = res.x[idx] if (auto_fit_copula and copula_type != "Fără") else copula_theta
        if auto_fit_copula and copula_type != "Fără": idx += 1
        fitted_phi = res.x[idx] if use_nbinom else 0.0

        matrix = self._generate_matrix(lambda_pure, mu_pure, rho_pure, pi_zero_pure, copula_type, fitted_theta, fitted_phi)
        
        top_4_prob = np.sum(np.sort(matrix.flatten())[-4:])
        flat_m = matrix.flatten()[matrix.flatten() > 0]
        
        return lambda_pure, mu_pure, rho_pure, pi_zero_pure, top_4_prob / (1.0 - top_4_prob + 1e-6), -np.sum(flat_m * np.log2(flat_m)), self.calculate_gini_index(matrix), self.calculate_top3_density(matrix), self.calculate_modal_skewness(lambda_pure, mu_pure), self.calculate_mvi(lambda_pure, mu_pure), matrix, fitted_theta, fitted_phi

    def decode_comparative(self, cs_open, cs_close, main_ah_open, main_ah_close, main_ou_open, main_ou_close, main_x_open=None, main_x_close=None, mode="Decoupled", copula_type="Fără", copula_theta=1.5, auto_fit_copula=False, use_nbinom=False):
        l_open_pre, m_open_pre, _, _, _, _, _, _, _, _, mat_open_pre, _, _ = self.extract_pure_xg(cs_open, main_ah_open, main_ou_open, main_x_open, mode, copula_type, copula_theta, auto_fit_copula, use_nbinom, z_mri_x=0.0)
        l_close_pre, m_close_pre, _, _, _, _, _, _, _, _, mat_close_pre, _, _ = self.extract_pure_xg(cs_close, main_ah_close, main_ou_close, main_x_close, mode, copula_type, copula_theta, auto_fit_copula, use_nbinom, z_mri_x=0.0)

        kl_pre = self.calculate_kl_divergence(mat_open_pre, mat_close_pre)
        jsd_pre = self.calculate_jsd(mat_open_pre, mat_close_pre)
        mri_pre = self.calculate_market_refractive_index(jsd_pre, main_ah_open['home_line'], main_ah_close['home_line'])
        fair_x_pre = 1.0 / max(self.calculate_draw_probability(mat_close_pre), 1e-5)
        x_gap_pre = abs(fair_x_pre - (main_x_close if main_x_close and main_x_close > 1.0 else fair_x_pre))
        
        z_mri_x = self.calculate_x_stress_z_score(kl_pre, x_gap_pre, mri_pre)

        l_open, m_open, r_open, pi_open, cs_ratio_open, ent_open, gini_open, top3_open, skew_open, mvi_open, mat_open, theta_open, phi_open = self.extract_pure_xg(cs_open, main_ah_open, main_ou_open, main_x_open, mode, copula_type, copula_theta, auto_fit_copula, use_nbinom, z_mri_x)
        l_close, m_close, r_close, pi_close, cs_ratio_close, ent_close, gini_close, top3_close, skew_close, mvi_close, mat_close, theta_close, phi_close = self.extract_pure_xg(cs_close, main_ah_close, main_ou_close, main_x_close, mode, copula_type, copula_theta, auto_fit_copula, use_nbinom, z_mri_x)

        delta_xg = (l_close + m_close) - (l_open + m_open)
        kl_div = self.calculate_kl_divergence(mat_open, mat_close)
        jsd_div = self.calculate_jsd(mat_open, mat_close)
        mri_index = self.calculate_market_refractive_index(jsd_div, main_ah_open['home_line'], main_ah_close['home_line'])
        ear_srp = self.calculate_erlang_adjusted_srp(jsd_div, main_ah_open['home_line'], main_ah_close['home_line'], l_close + m_close)
        biv_skew = self.calculate_bivariate_skewness_tensor(mat_close)
        tail_corr = self.calculate_analytical_tail_dependence(mat_close)
        shin_var = self.calculate_shin_alpha_variance(cs_close, main_ah_close, main_ou_close, main_x_close)

        u_x, v_y, fig_quiver = self.calculate_vector_field_flow(mat_open, mat_close)
        lap_map, max_curv, stiffness = self.calculate_surface_laplacian(mat_close)
        is_bimodal, num_peaks, peaks = self.calculate_topological_bimodal_index(mat_close)
        coskew_ha, regime_name, regime_desc = self.calculate_bivariate_moments(mat_close)
        exp_min, p15, p30 = self.calculate_first_passage_time(l_close, m_close)

        cmsi_kl_val = self.calculate_cross_market_stress_kl(mat_close_pre, mat_close, main_ou_close)
        apf_score, diag_asym, diag_curv = self.calculate_asymmetric_pressure_field(mat_close)
        
        raw_close_matrix = np.zeros((9, 9))
        for (h, a), odd in cs_close.items():
            if h <= 8 and a <= 8 and odd > 1.0: raw_close_matrix[h, a] = 1.0 / odd
        sum_raw = np.sum(raw_close_matrix)
        if sum_raw > 0: raw_close_matrix /= sum_raw

        vig_sq_mat, max_squeeze, squeezed_cells = self.calculate_cell_level_vig_squeeze(mat_close, raw_close_matrix)
        mod7_res = self.calculate_modulo_7_integrated_metrics(mat_open, mat_close, ent_close, gini_close, l_close + m_close)

        # RUN REVERSE ENGINEERING PE CENTRAL SERVER
        bookie_state = self.reverse_engineer_bookmaker_state(cs_close, main_ah_close, main_ou_close, main_x_close)

        fair_x_odd = 1.0 / max(self.calculate_draw_probability(mat_close), 1e-5)
        x_gap = abs(fair_x_odd - (main_x_close if main_x_close and main_x_close > 1.0 else fair_x_odd))
        stress_index = self.calculate_cross_market_stress_index(kl_div, shin_var, x_gap)

        scenario = "Scenariul D: Sharp Re-evaluation (Piață Recalibrată)"
        signal = "✅ PIAȚĂ ECHILIBRATĂ"
        explanation = f"Piața s-a recalibrat natural. xG Total s-a mutat cu {round(delta_xg, 2)} goluri."

        if stress_index > 0.75 or cmsi_kl_val > 0.65:
            scenario = "Scenariul S: Cross-Market Anomaly (Stress Elevat)"
            signal = "🚨 ALERTĂ MAXIMĂ: INEFIENȚĂ STRUCTURALĂ DE PIAȚĂ (TOOTHPRINTS DETECTED)"
            explanation = f"CMSI a atins {round(cmsi_kl_val, 2)}. Există o ruptură masivă între Correct Score, Handicap și 1X2."

        return {
            'open_l': round(l_open, 2), 'open_m': round(m_open, 2), 'open_xg': round(l_open+m_open, 2),
            'close_l': round(l_close, 2), 'close_m': round(m_close, 2), 'close_xg': round(l_close+m_close, 2),
            'delta_l': round(l_close - l_open, 2), 'delta_m': round(m_close - m_open, 2), 'delta_xg': round(delta_xg, 2),
            'cs_ratio_close': round(cs_ratio_close, 2), 'delta_cs': round(cs_ratio_close - cs_ratio_open, 2),
            'ent_close': round(ent_close, 2), 'delta_ent': round(ent_close - ent_open, 2),
            'gini_close': round(gini_close, 3), 'delta_gini': round(gini_close - gini_open, 3),
            'top3_close': round(top3_close, 1), 'delta_top3': round(top3_close - top3_open, 1),
            'skew_close': round(skew_close, 2), 'mvi_close': round(mvi_close, 2),
            'kl_div': round(kl_div, 4), 'jsd_div': round(jsd_div, 4), 'mri_index': round(mri_index, 4),
            'ear_srp': round(ear_srp, 4), 'biv_skew': round(biv_skew, 3),
            'tail_corr': round(tail_corr, 3), 'shin_var': round(shin_var, 5), 'x_gap': round(x_gap, 2),
            'stress_index': round(stress_index, 3), 'fitted_theta': round(theta_close, 2), 'fitted_phi': round(phi_close, 3),
            'scenario': scenario, 'signal': signal, 'explanation': explanation, 'matrix_close': mat_close,
            'fig_quiver': fig_quiver, 'lap_map': lap_map, 'max_curv': round(max_curv, 4), 'stiffness': round(stiffness, 4),
            'is_bimodal': is_bimodal, 'num_peaks': num_peaks, 'peaks': peaks,
            'coskew_ha': round(coskew_ha, 4), 'regime_name': regime_name, 'regime_desc': regime_desc,
            'exp_min': round(exp_min, 1), 'p15': round(p15, 1), 'p30': round(p30, 1),
            'cmsi_kl_val': round(cmsi_kl_val, 4), 'apf_score': round(apf_score, 4), 'diag_asym': round(diag_asym, 4),
            'vig_sq_mat': vig_sq_mat, 'max_squeeze': round(max_squeeze, 2), 'squeezed_cells': squeezed_cells,
            'mod7_res': mod7_res,
            'bookie_state': bookie_state
        }


# ==========================================
# STREAMLIT UI
# ==========================================

@st.cache_data(show_spinner="Calculare matrice, reverse engineering și detectare amprente...")
def run_cached_engine_decode(cs_open_input, cs_close_input, main_ah_open, main_ah_close, main_ou_open, main_ou_close, main_x_open, main_x_close, mode, copula_type, copula_theta, auto_fit_copula, use_nbinom):
    engine = PureMarketEngine9x9()
    return engine.decode_comparative(
        cs_open_input, cs_close_input, main_ah_open, main_ah_close, main_ou_open, main_ou_close, main_x_open, main_x_close, 
        mode=mode, copula_type=copula_type, copula_theta=copula_theta, 
        auto_fit_copula=auto_fit_copula, use_nbinom=use_nbinom
    )

st.title("🕵️ Pure Market Engine 9x9 (Optimized + Copula + Market Reverse Engineering)")

ah_options = [round(x, 2) for x in np.arange(-3.50, 3.75, 0.25)]
ou_options = [round(x, 2) for x in np.arange(1.25, 5.25, 0.25)]

cs_main = {
    (0,0): 12.0, (1,1): 6.8, (2,2): 13.5, (3,3): 50.0,
    (1,0): 9.0, (2,0): 15.0, (2,1): 11.0, (3,0): 35.0, (3,1): 22.0, (3,2): 28.0, (4,0): 80.0, (4,1): 65.0, (4,2): 70.0,
    (0,1): 8.5, (0,2): 10.5, (1,2): 8.35, (0,3): 24.0, (1,3): 15.0, (2,3): 20.0, (0,4): 60.0, (1,4): 50.0, (2,4): 55.0
}

st.sidebar.header("⚙️ Setări Motor Optimizare")
engine_mode = st.sidebar.radio("Mod Optimizare Solver:", ["Decoupled (Pure CS Matrix)", "Anchored (5.0 Weight)"])
use_nbinom = st.sidebar.checkbox("Activare Negative Binomial (Overdispersion)", value=False)
copula_type = st.sidebar.selectbox("Selectează Model Copula:", ["Fără", "Frank", "Gumbel", "Clayton"])
auto_fit_copula = st.sidebar.checkbox("Auto-Fit Optim Parametru Θ (Copula)", value=True)
copula_theta = 1.5

cs_open_input, cs_close_input = {}, {}

with st.sidebar.expander("📌 Correct Score OPEN", expanded=False):
    for score, default_odd in cs_main.items():
        cs_open_input[score] = st.number_input(f"Open {score[0]}-{score[1]}", value=default_odd, step=0.25, key=f"op_m_{score}")

with st.sidebar.expander("📌 Correct Score CLOSE", expanded=True):
    for score, default_odd in cs_main.items():
        cs_close_input[score] = st.number_input(f"Close {score[0]}-{score[1]}", value=default_odd, step=0.25, key=f"cl_m_{score}")

col_a1, col_a2 = st.sidebar.columns(2)
with col_a1:
    home_ah_line_op = st.selectbox("AH Gazde Open", ah_options, index=13, key="ah_line_op")
    home_ah_odd_op = st.number_input("Cotă AH Gazde Open", value=1.95, step=0.01, key="ah_h_op")
    away_ah_odd_op = st.number_input("Cotă AH Oaspeți Open", value=1.95, step=0.01, key="ah_a_op")
    main_ou_line_op = st.selectbox("O/U Open", ou_options, index=5, key="ou_line_op")
    main_ou_odd_op = st.number_input("Cotă Over Open", value=1.90, step=0.01, key="ou_o_op")
    under_ou_odd_op = st.number_input("Cotă Under Open", value=1.90, step=0.01, key="ou_u_op")
    main_x_odd_op = st.number_input("Cotă X Open", value=3.40, step=0.05, key="x_op")

with col_a2:
    home_ah_line_cl = st.selectbox("AH Gazde Close", ah_options, index=13, key="ah_line_cl")
    home_ah_odd_cl = st.number_input("Cotă AH Gazde Close", value=1.95, step=0.01, key="ah_h_cl")
    away_ah_odd_cl = st.number_input("Cotă AH Oaspeți Close", value=1.95, step=0.01, key="ah_a_cl")
    main_ou_line_cl = st.selectbox("O/U Close", ou_options, index=5, key="ou_line_cl")
    main_ou_odd_cl = st.number_input("Cotă Over Close", value=1.90, step=0.01, key="ou_o_cl")
    under_ou_odd_cl = st.number_input("Cotă Under Close", value=1.90, step=0.01, key="ou_u_cl")
    main_x_odd_cl = st.number_input("Cotă X Close", value=3.40, step=0.05, key="x_cl")

main_ah_open = {'home_line': home_ah_line_op, 'home_odd': home_ah_odd_op, 'away_odd': away_ah_odd_op}
main_ah_close = {'home_line': home_ah_line_cl, 'home_odd': home_ah_odd_cl, 'away_odd': away_ah_odd_cl}
main_ou_open = {'line': main_ou_line_op, 'over_odd': main_ou_odd_op, 'under_odd': under_ou_odd_op}
main_ou_close = {'line': main_ou_line_cl, 'over_odd': main_ou_odd_cl, 'under_odd': under_ou_odd_cl}

res = run_cached_engine_decode(
    cs_open_input, cs_close_input, 
    main_ah_open, main_ah_close, 
    main_ou_open, main_ou_close, 
    main_x_odd_op, main_x_odd_cl, 
    engine_mode, copula_type, copula_theta, 
    auto_fit_copula, use_nbinom
)

st.subheader("1. Metricile xG Pure & Structură (Shin Unmargined)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("xG Pur Gazde (λ)", f"{res['close_l']}", delta=f"{res['delta_l']} vs Open")
c2.metric("xG Pur Oaspeți (μ)", f"{res['close_m']}", delta=f"{res['delta_m']} vs Open")
c3.metric("xG Pur Total", f"{res['close_xg']}", delta=f"{res['delta_xg']} vs Open")
c4.metric("CS Ratio (Concentrare)", f"{res['cs_ratio_close']}", delta=f"{res['delta_cs']} vs Open")
c5.metric("Entropie Shannon (Haos)", f"{res['ent_close']}", delta=f"{res['delta_ent']} vs Open")

st.markdown("---")

# AFISARE REVERSE ENGINEERING SERVER CENTRAL
st.subheader("🖥️ Reverse Engineering Pe Serverul Central al Casei de Pariuri")
bs = res['bookie_state']
if bs:
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Model Copula Rulat", f"{bs['copula_type']}")
    rc2.metric("Parametru θ Copula", f"{bs['copula_theta']}")
    rc3.metric("AIC Score (Akaike)", f"{bs['aic_score']}")
    rc4.metric("Acuratețe Potrivire", f"{bs['fit_quality']}")
    
    st.info(
        f"**Vector de Stare Identificat (θ Server Central):**\n"
        f"- λ Gazde: `{bs['lambda_home']}` | μ Oaspeți: `{bs['mu_away']}`\n"
        f"- Corelație Dixon-Coles (ρ): `{bs['dixon_coles_rho']}` | Overdispersion Nbinom (ϕ): `{bs['nbinom_phi']}`"
    )

st.markdown("---")

st.subheader("🌊 Analiza Dinamică & Topologică")
tab_mod1, tab_mod2, tab_mod6, tab_mod7 = st.tabs([
    "Modulul 1: Vector Field Flow", 
    "Modulul 2: Surface Laplacian", 
    "🔥 Modulul 6: Market Toothprints",
    "🧪 Modulul 7: Time Decay & MEVI"
])

with tab_mod1:
    st.pyplot(res['fig_quiver'])

with tab_mod2:
    st.dataframe(np.round(res['lap_map'], 4), use_container_width=True)

with tab_mod6:
    t1, t2, t3 = st.columns(3)
    t1.metric("CMSI (KL Divergence)", f"{res['cmsi_kl_val']}")
    t2.metric("APF Score", f"{res['apf_score']}")
    t3.metric("Max Vig Squeeze", f"{res['max_squeeze']}x")
    st.dataframe(np.round(res['vig_sq_mat'], 2), use_container_width=True)

with tab_mod7:
    m7 = res['mod7_res']
    col_m7_1, col_m7_2, col_m7_3 = st.columns(3)
    col_m7_1.metric("Time Decay Drift", f"{m7['abs_drift']}")
    col_m7_2.metric("Decay Acceleration", f"{m7['decay_acc']}")
    col_m7_3.metric("MEVI Score", f"{m7['mevi_score']}", delta=f"Nivel {m7['vuln_level']}")

st.markdown("---")

st.subheader("Matricea Pură 9x9 Fără Marjă (%)")
st.dataframe(np.round(res['matrix_close'] * 100, 2), use_container_width=True)