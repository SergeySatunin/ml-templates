import numpy as np
import math

def bootstrap_estimate(data, rounds, stat, conf_level = 95, calc_bootstrap_t_ci = True):

    n = len(data)
    observed_stat = stat(data)

    boot_stats = []
    standard_errors = []

    for _ in range(rounds):
        bootsample = np.random.choice(data, size=n, replace=True)
        bootsample_stat = stat(bootsample)
        boot_stats.append(bootsample_stat)

        if calc_bootstrap_t_ci:
            new_boot_stats = []
            for _ in range(rounds):
                new_boot_sample = np.random.choice(bootsample, size=n, replace=True)
                new_bootsample_stat = stat(new_boot_sample)
                new_boot_stats.append(new_bootsample_stat)
            standard_errors.append(np.std(new_boot_stats))

    boot_stats = sorted(boot_stats)
    bootstrap_stat = stat(boot_stats)
    standard_error = np.std(boot_stats)

    low_percentile = (100 - conf_level) / 2
    up_percentile = 0.5 * (conf_level + 100)

    result = {
        'sample_stat' : observed_stat,
        'bootstrap_stat' : bootstrap_stat,
        'bootstrap_standard_error' : standard_error,
        'bootstrap_basic_ci': (2 * observed_stat - np.percentile(boot_stats, up_percentile),
                               2 * observed_stat - np.percentile(boot_stats, low_percentile)),
        'bootstrap_percentile_ci' : (np.percentile(boot_stats, low_percentile),
                                     np.percentile(boot_stats, up_percentile))
    }

    if calc_bootstrap_t_ci:
        bootstrap_t = []
        for i in range(len(boot_stats)):
            bootstrap_t.append((boot_stats[i] - bootstrap_stat) / standard_errors[i])

        bootstrap_t = sorted(bootstrap_t)
        result['bootstrap_t_ci'] = (observed_stat - standard_error*np.percentile(bootstrap_t, up_percentile),
                                    observed_stat - standard_error*np.percentile(bootstrap_t, low_percentile))

    return result

def bootstrap_mean_diff_estimate(control, treatment, rounds, conf_level = 95):

    n = len(treatment)
    m = len(control)

    treatment_mean = np.mean(treatment)
    control_mean = np.mean(control)
    observed_diff = treatment_mean - control_mean

    treatment_means = []
    control_means = []

    for _ in range(rounds):
        treatment_means.append(bootstrap_estimate(treatment, n, np.mean, 95, False)['bootstrap_stat'])
        control_means.append(bootstrap_estimate(control, m, np.mean, 95, False)['bootstrap_stat'])

    boot_means = np.array(treatment_means) - np.array(control_means)
    boot_means = sorted(boot_means)

    low_percentile = (100 - conf_level) / 2
    up_percentile = 0.5 * (conf_level + 100)

    result = {
        'sample_diff' : observed_diff,
        'bootstrap_diff' : np.mean(boot_means),
        'bootstrap_basic_ci': (2 * observed_diff - np.percentile(boot_means, up_percentile),
                               2 * observed_diff - np.percentile(boot_means, low_percentile)),
        'bootstrap_percentile_ci' : (np.percentile(boot_means, low_percentile),
                                     np.percentile(boot_means, up_percentile))
    }

    return result

# Efron, B.; Tibshirani, R. (1993). An Introduction to the Bootstrap. ISBN 0-412-04231-2.
def bootstrap_mean_t_test(control, treatment, rounds):

    def t_stat(x_mean, x_std, x_size, y_mean, y_std, y_size):
        return (x_mean - y_mean) / math.sqrt(pow(x_std, 2) / x_size + pow(y_std, 2) / y_size)

    n = len(treatment)
    m = len(control)

    pooled_sample = list(treatment) + list(control)

    treatment_mean = np.mean(treatment)
    control_mean = np.mean(control)
    pooled_mean = np.mean(pooled_sample)

    treatment_std = np.std(treatment)
    control_std = np.std(control)

    observed_t_stat = t_stat(treatment_mean, control_mean, treatment_std, control_std, n, m)

    treatment_shifted = np.array(treatment) - treatment_mean + pooled_mean
    control_shifted = np.array(control) - control_mean + pooled_mean

    boot_t_stats = []

    for _ in range(rounds):
        treatment_bootsample = np.random.choice(treatment_shifted, n, replace=True)
        control_bootsample = np.random.choice(control_shifted, m, replace=True)

        treatment_bootsample_mean = np.mean(treatment_bootsample)
        control_bootsample_mean = np.mean(control_bootsample)

        treatment_bootsample_std = np.std(treatment_bootsample)
        control_bootsample_std = np.std(control_bootsample)

        boot_t_stat = t_stat(treatment_bootsample_mean, control_bootsample_mean,
                             treatment_bootsample_std, control_bootsample_std,
                             n, m)

        boot_t_stats.append(boot_t_stat)

    cases = [x > observed_t_stat for x in boot_t_stats]
    p_value = sum(cases) / len(cases)

    return p_value