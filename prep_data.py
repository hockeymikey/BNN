from data_prep.data_utils import sort_veg_imgs_to_test, pickle_veg_data
from data_prep.img_extract import crisscross
from data_prep.lbl_utils import get_sorted_veg_from_excel

if __name__ == "__main__":
  while True:
    print('Can chain cmds(ex. esp, ep)')
    print('Extract imgs(e)\nSort Imgs(s)'
          '\nPixel Data(p)\nComplete(c)\nQuit(q)')
    inp = input('->: ')
    
    if inp == 'q':
      print('Good bye!')
      break
    
    valid = False
    
    if inp.__contains__('c'):
      valid = True
      
      crisscross()
      sort_veg_imgs_to_test()
      pickle_veg_data()
    else:
      if inp.__contains__('e'):
        valid = True
        crisscross()
      if inp.__contains__('s'):
        valid = True
        sort_veg_imgs_to_test()
      if inp.__contains__('p'):
        valid = True
        pickle_veg_data()
    if not valid:
      print('Invalid cmd.')