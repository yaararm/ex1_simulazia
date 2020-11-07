import numpy as np
import pandas as pd
from scipy import stats as st
import chaospy as ch

from scipy.stats import ks_2samp
# set the seed - can be only int in numpy but whatever
def setSeed(seed):
    np.random.seed(seed)


# region generators
# set 500 random values for each component by its distribution
def generateComponentRandomValues(num_of_samples, is_halton):
    samples = halton_randoms_generator(0, 1, num_of_samples * 2) if is_halton else uniform_random_generator(0, 1,
                                                                                                            num_of_samples * 2)
    half_samples = samples[:len(samples) // 2]

    res = {'blade': normal_random_generator(samples, 42000, 663),
           'gearbox': log_normal_random_generator(samples, 11, 1.2),
           # 'generator': st.weibull_max.rvs(c=1, loc=76000, scale=1.2, size=num_of_samples),
           'generator': weibull_random_generator(half_samples, 76000, 1.2),
           'yaw_system': gumbel_random_generator(half_samples, 65000, 370),
           'pitch_system': normal_random_generator(samples, 84534, 506),
           'brake_system': exponential_random_generator(half_samples, 120000),
           # 'lubrication': st.weibull_max.rvs(c=1, loc=66000, scale=1.3, size=num_of_samples),
           # 'electrical_system': st.weibull_max.rvs(c=1, loc=35000, scale=1.5, size=num_of_samples),
           'lubrication': weibull_random_generator(half_samples, 66000, 1.3),
           'electrical_system': weibull_random_generator(half_samples, 35000, 1.5),
           'frequency_converter': exponential_random_generator(half_samples, 45000)}
    return res


# get normal dist random values (mean,std)
def uniform_random_generator(a, b, num_of_samples):
    return np.random.uniform(a, b, num_of_samples).tolist()


def halton_randoms_generator(a, b, num_of_samples):
    # distribution = ch.J(ch.Uniform(0, 1), ch.Uniform(0, 1))
    # samples = distribution.sample(num_of_samples, rule="halton")
    samples = ch.distributions.sampler.sequences.halton.create_halton_samples(num_of_samples, dim=1, burnin=-1,
                                                                              primes=())
    return samples[0].tolist()


# get normal dist random values
def normal_random_generator(uniforms_samples, mean, std):
    normal_samples = [np.sqrt(-2 * np.log(u)) * np.sin(2 * np.pi * v) for u, v in
                      zip(uniforms_samples[0::2], uniforms_samples[1::2])]
    return [t * std + mean for t in normal_samples]


# get normal dist random values
def log_normal_random_generator(uniforms_samples, mean, std):
    normal_samples = [np.sqrt(-2 * np.log(u)) * np.sin(2 * np.pi * v) for u, v in
                      zip(uniforms_samples[0::2], uniforms_samples[1::2])]
    normal_cor = [x * std + mean for x in normal_samples]
    log_normal_samples = [np.exp(x) for x in normal_cor]
    return log_normal_samples


# get exponential dist random values
def exponential_random_generator(uniforms_samples, mean):
    return [-np.log(x) / (1 / mean) for x in uniforms_samples]


# get weibull dist random values
def weibull_random_generator(uniforms_samples, scale, shape):
    return [scale * ((-np.log(x)) ** (1 / shape)) for x in uniforms_samples]


# get gumbel maximum dist estimators (location,scale)
def gumbel_random_generator(uniforms_samples, location, scale):
    #return (np.random.gumbel(location, scale, uniforms_samples)).astype(int)
    return [location - scale * np.log(-np.log(x)) for x in uniforms_samples]


# get gumbel minimum dist estimators (location,scale)
def gumbel_l_random_generator(uniforms_samples, location, scale):
    return st.gumbel_l.rvs(location,scale,uniforms_samples)
    #return (np.random.gumbel(location,scale,uniforms_samples)).astype(int)
    #return [location - scale * np.log(-np.log(x)) for x in uniforms_samples]


# endregion


# region estimators
# get normal dist estimators (mean,std)
def normal_estimators(values):
    return st.norm.fit(values)


# get log_normal dist estimators (mean,std)
def log_normal_estimators(values):
    n = len(values)
    mean = sum([np.log(x) for x in values]) / n
    std = np.sqrt(sum([(np.log(x) - mean) ** 2 for x in values]) / n)
    return mean, std


# get exponential dist estimators (mean)
def exponential_estimators(values):
    return st.expon.fit(values)[1]


# get weibull dist estimators (scale,shape)
def weibull_estimators(values):
    # params = st.weibull_max.fit(values)
    # return params[1], params[2]
    n = len(values)
    sum_x = sum(values)
    sum_ln_x = sum([np.log(x) for x in values])
    sum_xln_x = sum([x * np.log(x) for x in values])
    shape = 1 / ((sum_xln_x / sum_x) - sum_ln_x / n)
    scale = (sum([x ** shape for x in values]) / n) ** (1 / shape)
    return scale, shape


# get gumbel maximum dist estimators (location,scale)
def gumbel_estimators(values):
    return st.gumbel_r.fit(values)


# get gumbel minimum dist estimators (location,scale)
def gumbel_l_estimators(values):
    return st.gumbel_l.fit(values)


# get dict of all components estimators
def getAllEstimators(values):
    res = {'blade_mean_est': normal_estimators(values['blade'])[0],
           'blade_std_est': normal_estimators(values['blade'])[1],
           'gearbox_mean_est': log_normal_estimators(values['gearbox'])[0],
           'gearbox_std_est': log_normal_estimators(values['gearbox'])[1],
           'generator_scale_est': weibull_estimators(values['generator'])[0],
           'generator_shape_est': weibull_estimators(values['generator'])[1],
           'yaw_system_location_est': gumbel_estimators(values['yaw_system'])[0],
           'yaw_system_scale_est': gumbel_estimators(values['yaw_system'])[1],
           'pitch_system_mean_est': normal_estimators(values['pitch_system'])[0],
           'pitch_system_std_est': normal_estimators(values['pitch_system'])[1],
           'brake_system_mean_est': exponential_estimators(values['brake_system']),
           'lubrication_scale_est': weibull_estimators(values['lubrication'])[0],
           'lubrication_shape_est': weibull_estimators(values['lubrication'])[1],
           'electrical_system_scale_est': weibull_estimators(values['electrical_system'])[0],
           'electrical_system_shape_est': weibull_estimators(values['electrical_system'])[1],
           'frequency_converter_mean_est': exponential_estimators(values['frequency_converter'])}

    return res


# endregion


# region df properties
def init_est_df():
    df = pd.DataFrame(
        columns=['sequence generator', 'seed', 'n', 'blade_mean_est', 'blade_std_est', 'gearbox_mean_est',
                 'gearbox_std_est',
                 'generator_scale_est', 'generator_shape_est',
                 'yaw_system_location_est', 'yaw_system_scale_est',
                 'pitch_system_mean_est', 'pitch_system_std_est', 'brake_system_mean_est',
                 'lubrication_scale_est', 'lubrication_shape_est',
                 'electrical_system_scale_est', 'electrical_system_shape_est',
                 'frequency_converter_mean_est'])
    return df


def init_conf_df():
    df = pd.DataFrame(
        columns=['sequence generator', 'a', 'type', 'blade_mean_est', 'blade_std_est', 'gearbox_mean_est',
                 'gearbox_std_est',
                 'generator_scale_est', 'generator_shape_est',
                 'yaw_system_location_est', 'yaw_system_scale_est',
                 'pitch_system_mean_est', 'pitch_system_std_est', 'brake_system_mean_est',
                 'lubrication_scale_est', 'lubrication_shape_est',
                 'electrical_system_scale_est', 'electrical_system_shape_est',
                 'frequency_converter_mean_est'])
    return df


def add_est_round_to_df(est_df, sequence_generator, seed, n, estimators_dict):
    estimators_dict['sequence generator'] = sequence_generator
    estimators_dict['seed'] = seed
    estimators_dict['n'] = n
    est_df = est_df.append(estimators_dict, ignore_index=True)
    return est_df


def add_conf_round_to_df(conf_df, sequence_generator, a, type, conf_dict):
    conf_dict['sequence generator'] = sequence_generator
    conf_dict['a'] = a
    conf_dict['type'] = type
    conf_df = conf_df.append(conf_dict, ignore_index=True)
    return conf_df


def save_df_to_csv(data_frame, file_name):
    data_frame.to_csv(file_name + '.csv', encoding='utf-8', index=False)


# endregion

#region fit_test

# def all_fit_tests(all_dis):
#     mins = all_dis["mins"]
#     all_test = {}
#     for dist_key, dist_value

#test whether the same distribution
def kolmogorov_smirnov_test(values,mindist):
    test = ks_2samp(mindist, values)
    print('stat=%.3f, p=%.3f' % (test.statistic, test.pvalue))
    if (test.pvalue <0.05):
        print("probably the same distribution")
    else:
        print("probably different distribution")

#test whether the variabla has dependency
def chi_square_test(values,mindist):
    # from scipy.stats import chi2_contingency
    # obs = np.array([values, mindist])
    # stat, p, dof, expected = chi2_contingency(obs)
    from scipy.stats import chisquare
    stat, p = chisquare(mindist,values)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably independent')
    else:
        print('Probably dependent')

# test if they have the same dist
def anderson_daring_test(values,name_of_dist):
    from scipy.stats import anderson
    result = anderson(values, name_of_dist)
    print('stat=%.3f' % (result.statistic))
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            print('Probably Gaussian at the %.1f%% level' % (sl))
        else:
            print('Probably not Gaussian at the %.1f%% level' % (sl))

# endregion

# main:

# init df
df_est = init_est_df()
df_conf = init_conf_df()

# region 1). default sequence generator
# 1.b - 500 randoms
setSeed(4)
randoms = generateComponentRandomValues(500, False)
estimators = getAllEstimators(randoms)
df_est = add_est_round_to_df(df_est, 'default', 4, 500, estimators)

# 1.c(1) - same seed, 500 randoms
randoms = generateComponentRandomValues(500, False)
estimators = getAllEstimators(randoms)
df_est = add_est_round_to_df(df_est, 'default', 4, 500, estimators)

# 1.c(2) - different seed, 500 randoms
setSeed(5)
randoms = generateComponentRandomValues(500, False)
estimators = getAllEstimators(randoms)
df_est = add_est_round_to_df(df_est, 'default', 5, 500, estimators)

# 1.c(3) - different seed, 1000 randoms
setSeed(6)
randoms = generateComponentRandomValues(1000, False)
estimators = getAllEstimators(randoms)
df_est = add_est_round_to_df(df_est, 'default', 6, 1000, estimators)

# 1.d -
# init dict to save all estimators from 100 iteration of random values
all_estimators = {'blade_mean_est': [],
                  'blade_std_est': [],
                  'gearbox_mean_est': [],
                  'gearbox_std_est': [],
                  'generator_scale_est': [],
                  'generator_shape_est': [],
                  'yaw_system_location_est': [],
                  'yaw_system_scale_est': [],
                  'pitch_system_mean_est': [],
                  'pitch_system_std_est': [],
                  'brake_system_mean_est': [],
                  'lubrication_scale_est': [],
                  'lubrication_shape_est': [],
                  'electrical_system_scale_est': [],
                  'electrical_system_shape_est': [],
                  'frequency_converter_mean_est': [],
                  }

# iter 100 times
for i in range(100):
    #print(i)
    setSeed(i)
    randoms = generateComponentRandomValues(500, False)
    round_est = getAllEstimators(randoms)
    for key, value in round_est.items():
        all_estimators[key].append(value)

# 1.d(1) - Confidence interval of max 10% and min 10% estimators a=0.1
conf_interval_top10 = {}
for key, value in all_estimators.items():
    all_estimators[key] = sorted(all_estimators[key])
    max_min_10 = all_estimators[key][0:10]
    max_min_10.extend(all_estimators[key][90:100])
    conf_interval = st.norm.interval(0.9, np.mean(max_min_10), np.std(max_min_10))
    conf_interval_top10[key] = ','.join([format(x, '.3f') for x in conf_interval])

df_conf = add_conf_round_to_df(df_conf, 'default', 0.9, 'min_max_10% estimators', conf_interval_top10)

# 1.d(2) - Confidence interval of all estimators (normal dist of estimators) a=0.1
conf_interval_all = {}
for key, value in all_estimators.items():
    conf_interval = st.norm.interval(0.9, np.mean(value), np.std(value))
    conf_interval_all[key] = ','.join([format(x, '.3f') for x in conf_interval])
df_conf = add_conf_round_to_df(df_conf, 'default', 0.9, 'all estimators', conf_interval_all)
# endregion


# region 1). helton sequence generator
# 1.b - 50 randoms
randoms = generateComponentRandomValues(50, True)
estimators = getAllEstimators(randoms)
df_est = add_est_round_to_df(df_est, 'halton', 'na', 50, estimators)

# 1.c(2) - different seed, 200 randoms
randoms = generateComponentRandomValues(200, True)
estimators = getAllEstimators(randoms)
df_est = add_est_round_to_df(df_est, 'halton', 'na', 200, estimators)

# 1.c(3) - different seed, 1000 randoms
randoms = generateComponentRandomValues(500, True)
estimators = getAllEstimators(randoms)
df_est = add_est_round_to_df(df_est, 'halton', 'na', 500, estimators)
# endregion


save_df_to_csv(df_est, 'estimators')
save_df_to_csv(df_conf, 'conf intervals')


################ question 2 ##################
allNumberDict = generateComponentRandomValues(500, False)
allNumberDF = pd.DataFrame(allNumberDict, columns = ['blade','gearbox','generator', 'yaw_system','pitch_system','brake_system','lubrication','electrical_system','frequency_converter'])
allNumberDF['Dmin'] = allNumberDF.min(axis=1)
allNumberDF.to_csv("allNumbers.csv")

#2.a
min_estimators={
           'normal_estimators_mean_est': normal_estimators(allNumberDF['Dmin'])[0],
           'normal_estimators_std_est': normal_estimators(allNumberDF['Dmin'])[1],
           'log_normal_mean_est': log_normal_estimators(allNumberDF['Dmin'])[0],
           'log_normal_std_est': log_normal_estimators(allNumberDF['Dmin'])[1],
           'weibull_estimators_scale_est': weibull_estimators(allNumberDF['Dmin'])[0],
           'weibull_estimators_shape_est': weibull_estimators(allNumberDF['Dmin'])[1],
           'gumbel_estimators_location_est': gumbel_estimators(allNumberDF['Dmin'])[0],
           'gumbel_estimators_scale_est': gumbel_estimators(allNumberDF['Dmin'])[1],
           'gumbel_l_estimators_location_est': gumbel_l_estimators(allNumberDF['Dmin'])[0],
           'gumbel_l_estimators_scale_est': gumbel_l_estimators(allNumberDF['Dmin'])[1],
           'exponential_estimators_mean_est': exponential_estimators(allNumberDF['Dmin'])}

print("all new Estimators:")
for key, value in min_estimators.items():
    print(key, ' : ', value)


#2.b
all_new_dist = {
    'normal': normal_random_generator(uniform_random_generator(0,1,1000),min_estimators['normal_estimators_mean_est'], min_estimators['normal_estimators_std_est']),
    'exponential':exponential_random_generator(uniform_random_generator(0,1,500),min_estimators['exponential_estimators_mean_est']),
    'Weibull':weibull_random_generator(uniform_random_generator(0,1,500),min_estimators['weibull_estimators_scale_est'],min_estimators['weibull_estimators_shape_est']),
    'log_normal':log_normal_random_generator(uniform_random_generator(0,1,1000),min_estimators['log_normal_mean_est'],min_estimators['log_normal_std_est']),
    'extreme_minimum':gumbel_random_generator(uniform_random_generator(0,1,500),min_estimators['gumbel_estimators_location_est'],min_estimators['gumbel_estimators_scale_est']),
    'extreme_maximum':gumbel_random_generator(uniform_random_generator(0,1,500),min_estimators['gumbel_l_estimators_location_est'],min_estimators['gumbel_l_estimators_scale_est'])
}
all_6_dist = pd.DataFrame(all_new_dist, columns=['normal','exponential','Weibull','log_normal','extreme_minimum','extreme_maximum',])
all_6_dist.to_csv("6dist.csv")


#fir test fpr normal
print()
print("--------------------compare with Normal dist-------------------")
print("kolmogorov-smirnov:")
kolmogorov_smirnov_test(all_6_dist['normal'],allNumberDF['Dmin'])
print("chi-square:")
chi_square_test(all_6_dist['normal'],allNumberDF['Dmin'])
print("anderson darling:")
anderson_daring_test(allNumberDF['Dmin'],'norm')
print()

print("--------------------compare with Expo dist-------------------")
print("kolmogorov-smirnov:")
kolmogorov_smirnov_test(all_6_dist['exponential'],allNumberDF['Dmin'])
print("chi-square:")
chi_square_test(all_6_dist['exponential'],allNumberDF['Dmin'])
print("anderson darling:")
anderson_daring_test(allNumberDF['Dmin'],'expon')
print()

print("--------------------compare with Weibull dist-------------------")
print("kolmogorov-smirnov:")
kolmogorov_smirnov_test(all_6_dist['Weibull'],allNumberDF['Dmin'])
print("chi-square:")
chi_square_test(all_6_dist['Weibull'],allNumberDF['Dmin'])
# print("anderson darling:")
# anderson_daring_test(allNumberDF['Dmin'],'expon')
print()


print("--------------------compare with log normal dist-------------------")
print("kolmogorov-smirnov:")
kolmogorov_smirnov_test(all_6_dist['log_normal'],allNumberDF['Dmin'])
print("chi-square:")
chi_square_test(all_6_dist['log_normal'],allNumberDF['Dmin'])
print("anderson darling:")
anderson_daring_test(allNumberDF['Dmin'],'logistic')
print()

print("--------------------compare with extreme minimum value dist-------------------")
print("kolmogorov-smirnov:")
kolmogorov_smirnov_test(all_6_dist['extreme_minimum'],allNumberDF['Dmin'])
print("chi-square:")
chi_square_test(all_6_dist['extreme_minimum'],allNumberDF['Dmin'])
print("anderson darling:")
anderson_daring_test(allNumberDF['Dmin'],'gumbel_l')
print()


print("--------------------compare with extreme maximum value dist-------------------")
print("kolmogorov-smirnov:")
kolmogorov_smirnov_test(all_6_dist['extreme_maximum'],allNumberDF['Dmin'])
print("chi-square:")
chi_square_test(all_6_dist['extreme_maximum'],allNumberDF['Dmin'])
print("anderson darling:")
anderson_daring_test(allNumberDF['Dmin'],'gumbel_r')
print()
