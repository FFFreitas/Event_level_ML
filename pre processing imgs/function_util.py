import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Wedge, RegularPolygon
from matplotlib.collections import PatchCollection
from AnalysisEvent import AnalysisEvent
import math
import os


class DetectorImage:
    ''' Class with functions to generate the abastract images'''
    shapescale = 0.3
    
    deta = 0.02
    dphi = 0.02
    
    etamin = -5.0
    etamax = 5.0

    phimin = -math.pi
    phimax = math.pi

    n_bins_eta = int( (etamax-etamin)/deta)
    n_bins_phi = int( (phimax-phimin)/dphi)

    @classmethod
    def set_parameters(cls, deta, dphi):
        cls.deta = deta
        cls.dphi = dphi
    @classmethod
    def set_eta_min_max(cls, etamin, etamax):
        cls.etamin = etamin
        cls.etamax = etamax
   
    def get_image_from_particles(self,collection,idx,xsize,ysize,dpiS,fc='k'):
        
        img = np.zeros( (self.n_bins_eta, self.n_bins_phi) , np.int16 )

        def find_bin(eta,phi):
            
            #print "Check ",phi," ",self.phimin," ",self.dphi
            etabin = int((eta-self.etamin)/self.deta)
            phibin = int((phi-self.phimin)/self.dphi)

            return (etabin,phibin)
            
        for track in collection:

            (etabin, phibin) = find_bin(track.Eta,track.Phi)

            #print (track.Eta,track.Phi)," ",(etabin, phibin)

            if hasattr(track,"PT"):
                try:
                    img[etabin,phibin] = img[etabin,phibin] + track.PT
                except IndexError:
                    img[0,0] = -999
            else:
                try:
                    img[etabin,phibin] = img[etabin,phibin] + track.ET
                except IndexError:
                    img[0,0] = -999


                    
    def get_image_from_event_ind(self,event,idx,xsize,ysize,dpiS,fc='k',tpy='sig'):
        ''' function to generate Individual(circles,squares,hexagos) abstract plots
        to be further use in the Autoencoders.
        take the events from a delphes root file as a first argument.
        second argument (idx) takes a index(int, preferable 0) to label the output images.
        third and fourth arguments defines the x,y sizes of the output figures in pixels.
        fifith argument defines the dpi.
        sixth arguments sets the facecolor of the plots.
        seventh argument sets the color of the lines for the polygons to draw.
        eight argument sets if the file is a signal or background'''
        
        '''to set the name for the signal and BG events, it can be changed by using the 
        class method set_abs_paths'''
        abspath_tt = 'ttbar_plots/' 
        abspath_ww = 'ww_plots/' 
                
        @classmethod
        def set_abs_paths(cls,new_path1,new_path2):
            cls.abspath_tt = new_path1
            cls.abspath_ww = new_path1
        paths_ww = []
        paths_tt = []
        shapes = ['circles','squares','hexagons']
        for i in shapes:
            tt_dirs = abspath_tt + i
            ww_dirs = abspath_ww + i
            if not os.path.exists(tt_dirs):
                os.makedirs(tt_dirs)
            if not os.path.exists(ww_dirs):
                os.makedirs(ww_dirs)
            paths_ww.append(ww_dirs)
            paths_tt.append(tt_dirs)
        
        line_with = (xsize/dpiS)*0.5
        rgb_color = []
        if fc is not 'k':
            rgb_color = ['r','g','b']

        fig1, ax1 = plt.subplots(subplot_kw=dict(xlim=(self.etamin,self.etamax),
                                 ylim=(self.phimin,self.phimax)),
                                 figsize=(xsize/dpiS,ysize/dpiS),
                                 dpi=dpiS,facecolor=fc)        

        fig2, ax2 = plt.subplots(subplot_kw=dict(xlim=(self.etamin,self.etamax),
                                 ylim=(self.phimin,self.phimax)),
                                 figsize=(xsize/dpiS,ysize/dpiS),
                                 dpi=dpiS,facecolor=fc)        

        fig3, ax3 = plt.subplots(subplot_kw=dict(xlim=(self.etamin,self.etamax),
                                 ylim=(self.phimin,self.phimax)),
                                 figsize=(xsize/dpiS,ysize/dpiS),
                                 dpi=dpiS,facecolor=fc)        
        
        ax1.axis('off')
        ax1.set_facecolor(fc)

        ax2.axis('off')
        ax2.set_facecolor(fc)

        ax3.axis('off')
        ax3.set_facecolor(fc)

        for el in event.EFlowTrack:
            coords = ( el.Eta, el.Phi )
            
            #print "Circle at ",coords, math.log(el.PT), " pT ",el.PT
            if fc is 'k':
                circle = Circle(coords, self.shapescale*math.log(el.PT),
                                fill=False,lw=line_with, color='w')
                ax1.add_artist(circle)

            elif fc is 'w':
                circle = Circle(coords, self.shapescale*math.log(el.PT),
                                fill=False,lw=line_with, color=rgb_color[0])
                ax1.add_artist(circle)
            
        for el in event.EFlowPhoton:
            coords = ( el.Eta, el.Phi )
            
            #print "Triangle at ",coords, math.log(el.ET), " ET ",el.ET
            if fc is 'k':            
                square = RegularPolygon(coords, 4, self.shapescale*math.log(el.ET),
                                        fill=False,lw=line_with, color='w')            

                ax2.add_artist(square)

            elif fc is 'w':            
                square = RegularPolygon(coords, 4, self.shapescale*math.log(el.ET),
                                        fill=False,lw=line_with, color=rgb_color[1])            

                ax2.add_artist(square)
                
        for el in event.EFlowNeutralHadron:
            coords = ( el.Eta, el.Phi )
            
            #print "Pentagon at ",coords, math.log(el.ET), " ET ",el.ET
            if fc is 'k':
                hexagon = RegularPolygon(coords, 6, self.shapescale*math.log(el.ET),
                                         fill=False,lw=line_with, color='w')

                ax3.add_artist(hexagon)

            elif fc is 'w':
                hexagon = RegularPolygon(coords, 6, self.shapescale*math.log(el.ET),
                                         fill=False,lw=line_with, color=rgb_color[2])

                ax3.add_artist(hexagon)
                    
        if tpy == 'sig':
            if fc is 'k':
                extent1 = ax1.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
                fig1.savefig(paths_ww[0] + '/plotBW_circle_{}.png'.format(idx),
                             facecolor=ax1.get_facecolor(),bbox_inches = extent1)
                plt.close()

                extent2 = ax2.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
                fig2.savefig(paths_ww[1] + '/plotBW_square_{}.png'.format(idx),
                             facecolor=ax2.get_facecolor(),bbox_inches = extent2)
                plt.close()

                extent3 = ax3.get_window_extent().transformed(fig3.dpi_scale_trans.inverted())
                fig3.savefig(paths_ww[2] + '/plotBW_hexagon_{}.png'.format(idx),
                             facecolor=ax3.get_facecolor(),bbox_inches = extent3)
                plt.close()

            elif fc is 'w':
                extent1 = ax1.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
                fig1.savefig(paths_ww[0] + '/plotcolor_circle_{}.png'.format(idx),
                             facecolor=ax1.get_facecolor(),bbox_inches = extent1)
                plt.close()

                extent2 = ax2.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
                fig2.savefig(paths_ww[1] + '/plotcolor_square_{}.png'.format(idx),
                             facecolor=ax2.get_facecolor(),bbox_inches = extent2)
                plt.close()

                extent3 = ax3.get_window_extent().transformed(fig3.dpi_scale_trans.inverted())
                fig3.savefig(paths_ww[2] + '/plotcolor_hexagons_{}.png'.format(idx),
                             facecolor=ax3.get_facecolor(),bbox_inches = extent3)
                plt.close()
            
        elif tpy == 'bg':
            if fc is 'k':
                extent1 = ax1.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
                fig1.savefig(paths_tt[0] + '/plotBW_circle_{}.png'.format(idx),
                             facecolor=ax1.get_facecolor(),bbox_inches = extent1)
                plt.close()

                extent2 = ax2.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())            
                fig2.savefig(paths_tt[1] + '/plotBW_square_{}.png'.format(idx),
                             facecolor=ax2.get_facecolor(),bbox_inches = extent2)
                plt.close()

                extent3 = ax3.get_window_extent().transformed(fig3.dpi_scale_trans.inverted())
                fig3.savefig(paths_tt[2] + '/plotBW_hexagon_{}.png'.format(idx),
                             facecolor=ax3.get_facecolor(),bbox_inches = extent1)
                plt.close()
                
            elif fc is 'w':
                extent1 = ax1.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
                fig1.savefig(paths_tt[0] + '/plotcolor_circle_{}.png'.format(idx),
                             facecolor=ax1.get_facecolor(),bbox_inches = extent1)
                plt.close()

                extent2 = ax2.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())            
                fig2.savefig(paths_tt[1] + '/plotcolor_square_{}.png'.format(idx),
                             facecolor=ax2.get_facecolor(),bbox_inches = extent2)
                plt.close()

                extent3 = ax3.get_window_extent().transformed(fig3.dpi_scale_trans.inverted())
                fig3.savefig(paths_tt[2] + '/plotcolor_hexagon_{}.png'.format(idx),
                             facecolor=ax3.get_facecolor(),bbox_inches = extent1)
                plt.close()

            
            
    def get_image_from_event(self,event,idx,xsize,ysize,dpiS,fc='k',tpy='sig'):
        ''' function to generate color or BW abstract plots to be further use in the Autoencoders.
        take the events from a delphes root file as a first argument.
        second argument (idx) takes a index(int, preferable 0) to label the output images.
        third and fourth arguments defines the x,y sizes of the output figures in pixels.
        fifith argument defines the dpi.
        sixth arguments sets the facecolor of the plots.
        seventh argument sets if the file is a signal or background'''

        abspath_tt = 'ttbar_plots/' 
        abspath_ww = 'ww_plots/' 
                
        @classmethod
        def set_abs_paths(cls,new_path1,new_path2):
            cls.abspath_tt = new_path1
            cls.abspath_ww = new_path1

        if not os.path.exists(abspath_tt):
            os.makedirs(abspath_tt)
        if not os.path.exists(abspath_ww):
            os.makedirs(abspath_ww)
        

        
        rgb_color = []
        line_with = (xsize/dpiS)*0.5
        if fc is not 'k':
            rgb_color = ['r','g','b']


        fig, ax = plt.subplots(subplot_kw=dict(xlim=(self.etamin,self.etamax),
                                               ylim=(self.phimin,self.phimax)),
                              figsize=(xsize/dpiS,ysize/dpiS),dpi=dpiS,facecolor=fc)        
        patches = []
        ax.axis('off')
        ax.set_facecolor(fc)

        for el in event.EFlowTrack:
            coords = ( el.Eta, el.Phi )
            
            #print "Circle at ",coords, math.log(el.PT), " pT ",el.PT

            if fc is 'k':
                circle = Circle(coords, self.shapescale*math.log(el.PT),
                            fill=False,lw=line_with, color='w')
            
            elif fc is 'w':
                circle = Circle(coords, self.shapescale*math.log(el.PT),
                            fill=False,lw=line_with, color=rgb_color[0])
                
            ax.add_artist(circle)
            
        for el in event.EFlowPhoton:
            coords = ( el.Eta, el.Phi )
            
            #print "Triangle at ",coords, math.log(el.ET), " ET ",el.ET
            if fc is 'k':
                square = RegularPolygon(coords, 4, self.shapescale*math.log(el.ET),
                                    fill=False,lw=line_with, color='w')
            else:
                pass

            if fc is 'w':
                square = RegularPolygon(coords, 4, self.shapescale*math.log(el.ET),
                                    fill=False,lw=line_with, color=rgb_color[1])
            
                
            ax.add_artist(square)
            
        for el in event.EFlowNeutralHadron:
            coords = ( el.Eta, el.Phi )
            
            #print "Pentagon at ",coords, math.log(el.ET), " ET ",el.ET
            if fc is 'k':
                hexagon = RegularPolygon(coords, 6, self.shapescale*math.log(el.ET),
                                     fill=False,lw=line_with, color='w')
            else:
                pass

            if fc is 'w':
                hexagon = RegularPolygon(coords, 6, self.shapescale*math.log(el.ET),
                                     fill=False,lw=line_with, color=rgb_color[2])
                    
            ax.add_artist(hexagon)
            
                    
        if tpy == 'sig':
            if fc is 'k':
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(abspath_ww + 'plotBW_{}.png'.format(idx),
                            bbox_inches = extent,facecolor=ax.get_facecolor())
                plt.close()
            elif fc is 'w':
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(abspath_ww + 'plotcolor_{}.png'.format(idx),
                            bbox_inches = extent,facecolor=ax.get_facecolor())
                plt.close()

        elif tpy == 'bg':
            if fc is 'k':
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(abspath_tt + 'plotBW_{}.png'.format(idx),
                            bbox_inches = extent,facecolor=ax.get_facecolor())
                plt.close()
            elif fc is 'w':
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(abspath_tt + 'plotcolor_{}.png'.format(idx),
                            bbox_inches = extent,facecolor=ax.get_facecolor())
                plt.close()
            