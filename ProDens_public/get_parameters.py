import numpy as np
from calculate_dataspace import Find_Density, Get_Parameterspace

input_density = float(input('Enter desired density: '))
input_thickness = float(input('Enter desired Layer Thickness: '))


dataspace = Get_Parameterspace()

parameter_set = Find_Density(input_density, input_thickness, dataspace)

print('\nDesired Density: {:.2f}[{:.2f}]%, Achieved Build Rate: {:.2f} cmm/s, Layerthickness: {:.2f} mm, '
      'Laserpower: {:.0f} W, Scanspeed: {:.0f} mm/s, Hatch Distance: {:.2f} mm'.format(input_density, parameter_set[4],
        parameter_set[5], parameter_set[0], parameter_set[1], parameter_set[2], parameter_set[3]))
