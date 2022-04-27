import scipy.stats as st
mean = 194
std = 11.2

low = 175
up = 225

print(f'The probably that the observations be between '
      f'{low} and {up} is {100*(st.norm(mean,std).cdf(up)-st.norm(mean,std).cdf(low)):.2f}')
