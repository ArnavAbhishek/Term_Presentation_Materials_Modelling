# Initialize lattice with random orientations (1 to Q)
# Set parameters: J, T, kB
# for step in range(total_MCS):
#     for n in range(number_of_sites):
#         i = random_site()
#         j = random_neighbor(i)
#         ΔE = energy_change_if_flip(i, j)
#         if ΔE <= 0:
#             flip_orientation(i, j)
#         else:
#             r = random_number(0,1)
#             if r < exp(-ΔE / (kB*T)):
#                 flip_orientation(i, j)
#     measure_grain_size()
