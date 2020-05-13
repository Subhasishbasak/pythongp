class Mean():
    '''
    class defining mean
    '''
    def __init__(self):

       self.mean_type = None

    def get_mean_type(self):
        return self.mean_type

    def construct(self, *arg):
       '''
        the function takes the following argument:
            -> mean_type
        If no arguments are given, It enables the user to specify the mean type
        Available mean types are displayed on the console
       '''

       try:

            self.mean_type = arg[0]

       except IndexError:

            while (self.mean_type not in ['Zero', 'Constant', 'Sample', 'None']):

                if self.mean_type is None:

                    self.mean_type = input(
                        ' Specify "Sample" for the sample mean\n "Zero" for zero mean\n "Constant" for constant mean\n Zero for GPy\n\n Enter mean function: ')

                else:

                    self.mean_type = input('Enter correct mean type : ')
       return self

       
        