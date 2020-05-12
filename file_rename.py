
# Pythono3 code to rename multiple  
# files in a directory or folder 
  
# importing os module 
import os 
  
# Function to rename multiple files 
def main(): 
  
    for count, filename in enumerate(os.listdir("C:\\Users\\Mehmet\\Desktop\\ANN\\img_dataset\\fruits\\fruits-360\\Test\\")): 
        dst = "{}_{}".format(str(count), filename)
        src = 'C:\\Users\\Mehmet\\Desktop\\ANN\\img_dataset\\fruits\\fruits-360\\Test\\' + filename 
        dst = 'C:\\Users\\Mehmet\\Desktop\\ANN\\img_dataset\\fruits\\fruits-360\\Test\\' + dst 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 