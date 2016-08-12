import pandas as pd
import numpy as np
from datetime import *
import gpstk
import utm
import sys
import matplotlib.pyplot as plt
import warnings
from bokeh.plotting import *
from bokeh.charts import *
from bokeh.models import *
from motionless import CenterMap, DecoratedMap, LatLonMarker
from sklearn.naive_bayes import GaussianNB
import urllib
from skimage import io
from joblib import Parallel, delayed

import bokeh
import time, math
from bokeh.tile_providers import *
import pyproj
from sklearn.grid_search import GridSearchCV

warnings.filterwarnings('ignore')

colors=["red", "blue", "green", "brown", "black", "pink", "orange"]
secsInWeek = 604800
secsInDay = 86400
gpsEpoch = (1980, 1, 6, 0, 0, 0)  # (year, month, day, hh, mm, ss)


def load_augergps_data(fname):
    d=pd.read_csv(fname, delimiter=",")
    d=d[d["northing"]<8e8]
    d=d[d["easting"]<5.2e7]
#    d=d[d["stationid"]<2000]
    d=d[d["gpstime"]>900000000]
    return d

def load_augergps_raw(fname):
    with open(fname, 'r') as f:
        xyzt = [re.search("StationId=(\S+).+X=(\S+)\s+Y=(\S+)\s+Z=(\S+)\s+CurrentTime=(\S+)",line) for line in f]
        xyzt = [i.groups() for i in xyzt if i is not None]
        d = pd.DataFrame(np.array(xyzt).astype(int), columns=["stationid", "northing", "easting", "height", "gpstime"])
    f.close()
    return d

def load_campaigns(campaigns, datadir="data"):
    stations = {}
    campaign_data = {}
    for cid in np.sort(campaigns.keys()):
        campaign = campaigns[cid]
        d = load_campaign(campaign, datadir=datadir)
        print cid, gpssecs_to_utc(np.min(d.gpstime)), "->", gpssecs_to_utc(np.max(d.gpstime)), 
        stations[cid] = get_avg_station_positions(d)
        campaign_data[cid] = d
        print
    return stations, campaign_data

def latlng_to_meters(lat, lng):
    origin_shift = 2 * np.pi * 6378137 / 2.0
    mx = lng * origin_shift / 180.0
    my = np.log(np.tan((90 + lat) * np.pi / 360.0)) / (np.pi / 180.0)
    my = my * origin_shift / 180.0
    return mx, my

# utc = 1980-01-06UTC + (gps - (leap_count(2014) - leap_count(1980)))
def gpssecs_to_utc(seconds):
    utc = datetime(1980, 1, 6) + timedelta(seconds=int(seconds) - (35 - 19))
    return utc

def gpssecs_to_gpstktime(secs):
    ut = gpssecs_to_utc(secs)
    vt = gpstk.CivilTime()
    vt.day=ut.day
    vt.year=ut.year
    vt.month=ut.month
    vt.hour = ut.hour
    vt.minute = ut.minute
    vt.second = ut.second
    vt.setTimeSystem(gpstk.TimeSystem(gpstk.TimeSystem.GPS))
    return vt.toCommonTime()


def gpstktime_to_gpssecs(t):
    return t.getDays()*60*60*24-211182767984

def utc_to_gpssecs(t):
    week, sow, day, sod = gpsFromUTC(t.year, t.month, t.day, t.hour, t.minute, t.second, leapSecs=16)
    return week*secsInWeek + sow

def gpssecs_to_gpsday(t):
    dw = gpstk.GPSWeekSecond(gpssecs_to_gpstktime(t))
    return dw.getSOW()/(60*60*24) + dw.getWeek()*7   

def gpsFromUTC(year, month, day, hour, min, sec, leapSecs=14):
    """converts UTC to: gpsWeek, secsOfWeek, gpsDay, secsOfDay


    from: https://www.lsc-group.phys.uwm.edu/daswg/projects/glue/epydoc/lib64/python2.4/site-packages/glue/gpstime.py

    a good reference is:  http://www.oc.nps.navy.mil/~jclynch/timsys.html

    This is based on the following facts (see reference above):

    GPS time is basically measured in (atomic) seconds since 
    January 6, 1980, 00:00:00.0  (the GPS Epoch)
    
    The GPS week starts on Saturday midnight (Sunday morning), and runs
    for 604800 seconds. 

    Currently, GPS time is 13 seconds ahead of UTC (see above reference).
    While GPS SVs transmit this difference and the date when another leap
    second takes effect, the use of leap seconds cannot be predicted.  This
    routine is precise until the next leap second is introduced and has to be
    updated after that.  

    SOW = Seconds of Week
    SOD = Seconds of Day

    Note:  Python represents time in integer seconds, fractions are lost!!!
    """

    secFract = sec % 1
    epochTuple = gpsEpoch + (-1, -1, 0)
    t0 = time.mktime(epochTuple)
    t = time.mktime((year, month, day, hour, min, sec, -1, -1, 0)) 
    # Note: time.mktime strictly works in localtime and to yield UTC, it should be
    #       corrected with time.timezone
    #       However, since we use the difference, this correction is unnecessary.
    # Warning:  trouble if daylight savings flag is set to -1 or 1 !!!
    t = t + leapSecs   
    tdiff = t - t0
    gpsSOW = (tdiff % secsInWeek)  + secFract
    gpsWeek = int(math.floor(tdiff/secsInWeek)) 
    gpsDay = int(math.floor(gpsSOW/secsInDay))
    gpsSOD = (gpsSOW % secsInDay) 
    return (gpsWeek, gpsSOW, gpsDay, gpsSOD)

def read_rinex(obsfile, navfile, mintime=None, maxtime=None, min_observation_period=None):
    navHeader, navData = gpstk.readRinex3Nav(navfile)
        
    # setup ephemeris store to look for satellite positions
    bcestore = gpstk.GPSEphemerisStore()
    for navDataObj in navData:
        ephem = navDataObj.toGPSEphemeris()
        bcestore.addEphemeris(ephem)
    bcestore.SearchNear()
    navData.close()

    obsHeader, obsData = gpstk.readRinex3Obs(obsfile)
    dsats = {}
    dposs = []
    rec_pos = gpstk.Position(obsHeader.antennaPosition[0], obsHeader.antennaPosition[1], obsHeader.antennaPosition[2])
    obs_types = np.array([i for i in obsHeader.R2ObsTypes])
    L1_idx = np.where(obs_types=="L1")[0][0]
    L2_idx = np.where(obs_types=="L2")[0][0]
    P1_idx = np.where(obs_types=="C1")[0][0]
    P2_idx = np.where(obs_types=="P2")[0][0]
    ipp_height        =  350000
    earth_mean_radius = 6371000
    print "index for L1, L2, P1, P2:", L1_idx, L2_idx, P1_idx, P2_idx
    invalid_ephems = 0
    last_gtime = -np.inf
    c=0
    for obsObject in obsData:
        gtime = gpstktime_to_gpssecs(obsObject.time)
        if min_observation_period is not None and (gtime - last_gtime)<min_observation_period:
            continue
        last_gtime = gtime
        if (mintime is None or gtime>=mintime) and (maxtime is None or gtime<=maxtime):
            prnList = []
            rangeList = []    
            noTropModel = gpstk.ZeroTropModel()

            for satID, datumList in obsObject.obs.iteritems():
                if satID.system==satID.systemGPS:
                    if not satID.id in dsats.keys():
                        dsats[satID.id]=[]
                    try:
                        eph   = bcestore.findEphemeris(satID, obsObject.time)
                        svXvt = eph.svXvt(obsObject.time)
                    except gpstk.InvalidRequest:
                        invalid_ephems += 1
                        continue
                           
                    elev  = rec_pos.elvAngle(svXvt.getPos())
                    azim  = rec_pos.azAngle(svXvt.getPos())
                    pp    = rec_pos.getIonosphericPiercePoint(elev, azim, ipp_height)
                    ipp_lat, ipp_lon, _ = ecef2lla(pp[0], pp[1], pp[2], isradians=False)

                    P1 = obsObject.getObs(satID, P1_idx).data
                    P2 = obsObject.getObs(satID, P2_idx).data
                    L1 = obsObject.getObs(satID, L1_idx).data
                    L2 = obsObject.getObs(satID, L2_idx).data

                    iono_delay_P = 1.0 / (1.0 - gpstk.GAMMA_GPS) * ( P1 - P2 )
                    iono_delay_L = 1.0 / (1.0 - gpstk.GAMMA_GPS) * ( L2 - L1 )

                    iono_delay_P = np.nan if iono_delay_P<0 else iono_delay_P

                    me = np.sqrt(1-(np.cos(elev*np.pi/180)/(1+ipp_height/earth_mean_radius))**2)
                    vtec_P = iono_delay_P*me
                    vtec_L = iono_delay_L*me
                        
                    dsats[satID.id]+= [[satID.id, P1,P2,L1,L2, elev, azim, 
                                            ipp_lat, ipp_lon,
                                            iono_delay_P, iono_delay_L, 
                                            vtec_P, vtec_L,
                                            gtime]]
                    prnList.append(satID)
                    rangeList.append(P1) 
            if len(prnList)>=4:
                raimSolver = gpstk.PRSolution2()
                satVector = gpstk.seqToVector(prnList, outtype='vector_SatID')
                rangeVector = gpstk.seqToVector(rangeList)
                # compute position
                raimSolver.RAIMCompute(obsObject.time, satVector, rangeVector, bcestore, noTropModel)   
                computed_pos = np.array([raimSolver.Solution[0], raimSolver.Solution[1], raimSolver.Solution[2]])
                
                lat, lon, alt = ecef2lla(raimSolver.Solution[0], raimSolver.Solution[1], raimSolver.Solution[2], isradians=False)
                mlat, mlon = latlng_to_meters(lat,lon)
                dposs.append([gtime, lat, lon, alt, mlat, mlon, raimSolver.Solution[0], raimSolver.Solution[1], raimSolver.Solution[2]])

    if invalid_ephems!=0:
        print "WARNING, ignored", invalid_ephems, "observations due to not found ephemeris"
    obsData.close()
    cols = ["PRN", "P1", "P2", "L1", "L2", "elev", "azim", "ipp_lat", "ipp_lon", "iono_delay_P", "iono_delay_L", "vtec_P", "vtec_L", "gpstime"]
    for i in dsats.keys():
        if len(dsats[i])>0:
            dsats[i] = pd.DataFrame(np.array(dsats[i]), columns=cols)
            dsats[i] = dsats[i].dropna()
        else:
            del dsats[i]
            print "WARNING, ignoring PRN",i,"with no observations"
    if len(dposs)>0:
        dposs = pd.DataFrame(np.array(dposs), columns=["gpstime", "lat", "lon", "mlat", "mlon", "alt", "X", "Y", "Z"])
    else:
        dposs = None
    return dsats, dposs

def smooth(data, window_size=10, crop=True):
    dd = data.copy()
    for i in range(window_size,len(data)-window_size):
        dd[i] = np.mean(data[i-window_size:i+window_size])
    if crop:
        dd = dd[window_size:-window_size]
    return dd

def smooth_col(c, colname, window_size=10, crop=True):
    swindow = 10
    dd = smooth(c[colname].copy().as_matrix(), window_size, crop=False)
    c["s_"+colname] = dd
    if crop:
        c = c.iloc[window_size:-window_size]
    return c

def discretize_column(d, colname, bin_size):
    binned_colname = "binned_"+colname
    if binned_colname in d.columns:
        d = d.drop(binned_colname,1)
    d = d.join(pd.DataFrame((d[colname]/bin_size).as_matrix().astype(int)*bin_size, columns=[binned_colname], index=d.index))
    return d

def get_avg_station_positions(d):
    stations = np.unique(d["stationid"]).astype(int)
    ns, es, hs = {}, {}, {}
    nd, ed, hd = {}, {}, {}
    nss, ess, hss = {}, {}, {}
    dp = {}
    nobs = {}
    ids = []
    for s in stations:
        if s%100==0:
            print s,
        ds = d[d["stationid"]==s]
        ids.append(int(s))
        ns[s] = np.mean(ds["northing"])
        es[s] = np.mean(ds["easting"])
        hs[s] = np.mean(ds["height"])
        nss[s] = np.std(ds["northing"])
        ess[s] = np.std(ds["easting"])
        hss[s] = np.std(ds["height"])
        hd[s] = np.sum((ds.height.iloc[1:].as_matrix()-ds.height.iloc[:-1].as_matrix())==0)*1./(len(ds)-1)
        ed[s] = np.sum((ds.easting.iloc[1:].as_matrix()-ds.easting.iloc[:-1].as_matrix())==0)*1./(len(ds)-1)
        nd[s] = np.sum((ds.northing.iloc[1:].as_matrix()-ds.northing.iloc[:-1].as_matrix())==0)*1./(len(ds)-1)
        
        dp[s] = np.sum(np.sqrt((ds.height.iloc[1:].as_matrix()-ds.height.iloc[:-1].as_matrix())**2 +  \
                               (ds.easting.iloc[1:].as_matrix()-ds.easting.iloc[:-1].as_matrix())**2 + \
                               (ds.northing.iloc[1:].as_matrix()-ds.northing.iloc[:-1].as_matrix())**2)==0)*1./(len(ds)-1)
        
        nobs[s] = len(ds)

    s = pd.DataFrame(np.array([ids, nobs.values(), es.values(), ns.values(), hs.values(), 
                               ess.values(), nss.values(), hss.values(), 
                               dp.values(),]).T, 
                     columns=["stationid", "observations", "easting", "northing", "height", 
                              "easting std", "northing std", "height std", 
                              "zero delta pct"], 
                     index=ids)
    s.stationid    = s.stationid.as_matrix().astype(int)
    s.observations = s.observations.as_matrix().astype(int)

    s = pd.concat([s, pd.DataFrame([utm.to_latlon(i[1]["easting"]/100, i[1]["northing"]/100, 19, "H") for i in s.iterrows()], 
                            columns=["lat", "long"], index=s.index)], axis=1)
    s = pd.concat([s, pd.DataFrame([latlng_to_meters(i[1]["lat"], i[1]["long"]) for i in s.iterrows()],
                 columns=["wlat", "wlong"], index=s.index)], axis=1)
    return s


def plot_sample_stations(campaign_data, stations, campaign):
    i=campaign
    d = campaign_data[i]
    obs = list(stations[i].observations.as_matrix())
    mode = int(max(set(obs), key=obs.count))

    plt.figure(figsize=(12,5))
    manyobs_stations = stations[i].iloc[np.argsort(-stations[i].observations)[:5]]
    fewobs_stations  = stations[i][stations[i].observations < 100][:5]
    regular_stations = stations[i][stations[i].observations==mode].iloc[:5]

    for s in regular_stations.iterrows():
        s2 = d[d.stationid==s[1].stationid]
        plt.scatter(s2.gpstime, s2.height, alpha=0.5, color="red", s=1, 
                    label="station "+str(int(s[1].stationid))+",  "+str(s[1].observations)+" obs")

    for s in manyobs_stations.iterrows():
        s2 = d[d.stationid==s[1].stationid]
        plt.scatter(s2.gpstime, s2.height, alpha=0.1, color="blue", s=1, 
                    label="station "+str(int(s[1].stationid))+",  "+str(s[1].observations)+" obs")
    for s in fewobs_stations.iterrows():
        s2 = d[d.stationid==s[1].stationid]
        plt.scatter(s2.gpstime, s2.height, alpha=1., color="green", s=10, 
                    label="station "+str(int(s[1].stationid))+",  "+str(s[1].observations)+" obs")
    plt.xlabel("gpstime")
    plt.ylabel("height")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    plt.title("CAMPAIGN "+campaign+ ", $mode$ for nb observations="+str(mode)+", nb of stations with "+str(mode)+" observations "+str(np.sum(stations[i].observations==mode)))
    plt.xlim(np.min(d.gpstime), np.max(d.gpstime))

def load_campaign(files, remove_fixed_observations=False, datadir="data"):
    d = None
    for i in files:
        dtmp = load_augergps_data(datadir+"/"+i)
        d = dtmp if d is None else d.append(dtmp, ignore_index=True)
    if remove_fixed_observations:
        d = add_pos_delta(d)
        d = d[d["pos delta"]!=0]
    return d

def get_common_stations(stations):
    common_ids = np.unique(stations[stations.keys()[0]].stationid)
    for i in stations.keys():
        ids = np.unique(stations[i].stationid).astype(int)
        common_ids = list(set(common_ids) & set(ids))
    return common_ids

def remove_unshared_stations(stations):
    common_ids = get_common_stations(stations)
    print "there are", len(common_ids), "common stations"

    print "removing stations not present in all campaings"
    for i in np.sort(stations.keys()):
        idx=[(np.in1d(stations[i]["stationid"], common_ids))][0]
        print "  ", i, "removed", len(stations[i])-np.sum(idx), "stations"
        stations[i] = stations[i][idx]

    return stations

def get_campaign_meanposition_comparison(stations):
    mean_positions = [[i, np.mean(stations[i]["easting"]), np.mean(stations[i]["northing"]), np.mean(stations[i]["height"])] 
                 for i in np.sort(stations.keys())]

    r = {}
    for i in mean_positions:
        r[i[0]] = {"easting": i[1], "northing": i[2], "height": i[3]}

    rd = pd.DataFrame(columns=np.sort(r.keys()), index=np.sort(r.keys()))
    for i in r.keys():
        for j in r.keys():
            pi = np.array([r[i]["easting"], r[i]["northing"], r[i]["height"]])
            pj = np.array([r[j]["easting"], r[j]["northing"], r[j]["height"]])
            rd.loc[i][j]=np.linalg.norm(pi-pj)
    return rd

def get_position_errors(d, stations):
    
    stations_dict = {}
    for s in stations.stationid:
        stations_dict[int(s)] = stations[stations.stationid==s][["northing", "easting", "height"]].as_matrix()[0]    
    dtimes = {}

    print "computing positioning errors for", len(d), "items ... ",
    sys.stdout.flush()
    c=1
    large_errors = 0
    for index, obs in d.iterrows():
        if c%10000==0:
            print c,
            sys.stdout.flush()
        c+=1
        t = obs["binned_gpstime"]
        st = obs["stationid"]
        if not t in dtimes.keys():
            dtimes[t] = dict(stationid=[], station_pos=[], gps_pos=[], pos_error=[])

        station_pos = stations_dict[st]
        gps_pos     = np.array([obs.northing, obs.easting, obs.height]).astype(float)
        pos_error   = station_pos - gps_pos

        dtimes[t]["stationid"].append(st)
        dtimes[t]["station_pos"].append(station_pos)
        dtimes[t]["gps_pos"].append(gps_pos)
        dtimes[t]["pos_error"].append(pos_error)
        
        
    print "\nbuilding dataframe ... ",
    sys.stdout.flush()
    dt = {}
    for t in dtimes.keys():
        lt = []
        for i in range(len(dtimes[t]["gps_pos"])):
            lt.append(dtimes[t]["stationid"][i])
            lt += list(dtimes[t]["gps_pos"][i])
            lt += list(dtimes[t]["station_pos"][i])
            lt += list(dtimes[t]["pos_error"][i])
            pe = dtimes[t]["pos_error"][i]
            lt.append(np.sqrt(pe[0]**2+pe[1]**2))
        lt = np.array(lt).reshape(len(dtimes[t]["gps_pos"]),11)
        dt[t] = pd.DataFrame(lt, columns=["stationid", "gps_n", "gps_e", "gps_h", 
                                          "station_n", "station_e", "station_h", 
                                          "error_n", "error_e", "error_vert", "error_hor"])
    dtimes = dt
    print "done"
    return dtimes

def plot_error_field(pos_errors, field, ylim=(-1000,1000)):
    sorted_times = np.sort(pos_errors.keys())    
    h_errors = [(np.mean(pos_errors[i][field]), np.std(pos_errors[i][field])) for i in sorted_times]
    max_errs = [np.max(pos_errors[i][field])  for i in sorted_times]
    min_errs = [np.min(pos_errors[i][field])  for i in sorted_times]
    plt.figure(figsize=(12,2))
    plt.plot(sorted_times, [i[0] for i in h_errors], label="avg")
    plt.fill_between(sorted_times,[i[0]-i[1] for i in h_errors], [i[0]+i[1] for i in h_errors] , 
                     alpha=0.2, color="yellow", label="std")
    plt.xlim(np.min(pos_errors.keys())-100, np.max(pos_errors.keys())+100)
    plt.xlabel("time")
    plt.ylabel(field)
    plt.plot(sorted_times, max_errs, color="red", label="max")
    plt.plot(sorted_times, min_errs, color="red", label="min")
    plt.legend()
    plt.ylim(ylim)

def plot_observations_stations_per_time(pos_errors):
    sorted_times = np.sort(pos_errors.keys())    
    sdata = [len(pos_errors[i])*1./len(np.unique(pos_errors[i]["stationid"])) for i in sorted_times]
    plt.figure(figsize=(12,1))
    plt.scatter( np.sort(pos_errors.keys()), sdata, c="blue", alpha=.5)
    plt.xlabel("time")
    plt.ylabel("ratio")
    plt.title("# observations / # stations per time point")
    plt.xlim(np.min(pos_errors.keys())-100, np.max(pos_errors.keys())+100)
    sdata = [len(np.unique(pos_errors[i]["stationid"])) for i in sorted_times]
    plt.figure(figsize=(12,1))
    plt.scatter( sorted_times, sdata, c="blue", alpha=.5)
    plt.ylabel("# stations")
    plt.xlabel("time")
    plt.title("# stations per time point")
    plt.xlim(np.min(pos_errors.keys())-100, np.max(pos_errors.keys())+100)
    plt.plot(sorted_times, np.ones(len(pos_errors))*1570, color="red", alpha=0.5, label="1570 stations")
    plt.legend()


def show_stations(stations, color="red", hold_show=False, fig=None):

    source = ColumnDataSource(data=dict(wlat = np.array(stations['wlat']), wlong=np.array(stations['wlong']), stationid=np.array(stations['stationid'])))
    hover = HoverTool(tooltips=[ ("id", "(@stationid)")])


    tools = [WheelZoomTool(), BoxZoomTool(),  ResetTool(), PanTool(), BoxSelectTool() ]
    tools.append(hover)
    openmap_url = 'http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png'
    otile_url = 'http://otile1.mqcdn.com/tiles/1.0.0/sat/{Z}/{X}/{Y}.jpg'
    OTILE = WMTSTileSource(url=otile_url)
    fig = figure(tools=tools, plot_width=900,plot_height=600) if fig is None else fig
    fig.circle("wlat", "wlong", source=source, color=color, alpha=0.5)
    fig.add_tile(OTILE)
    fig.axis.visible = False
    if not hold_show:
        show(fig)
    
    return fig
    
    
def add_pos_delta(d):
    dr = None
    c=1
    for i in np.unique(d.stationid):
        if c%100==0:
            print c,
        c+=1
        ds = d[d.stationid==i]
        ll = [0]+list(np.sqrt((ds[1:].northing.as_matrix() - ds[:-1].northing.as_matrix())**2  \
                             +(ds[1:].easting.as_matrix() - ds[:-1].easting.as_matrix())**2\
                             +(ds[1:].height.as_matrix() - ds[:-1].height.as_matrix())**2))
        ll = pd.DataFrame(ll, index=ds.index, columns=["pos delta"])
        dr = ll if dr is None else dr.append(ll)
    dr = dr.join(d)
    return dr


def plot_stations_fields(stations, field1, field2):
    fig=plt.figure(figsize=(15,3.5))
    c=1
    for i in np.sort(stations.keys()):
        fig.add_subplot(2,3,c)
        plt.scatter(stations[i][field1], stations[i][field2], alpha=0.5)    
        plt.xlim(0,4000)
        plt.title(i)
        if c <=2 : plt.xticks([])
        else: plt.xlabel(field1)
        if c==1 or c==4: plt.ylabel(field2)
        c+=1
        
def detail_station(sid, stations):
    s=stations

    k = np.sort(s.keys())
    easts, norths = [], []
    for i in range(len(k)):
        c = s[k[i]]
        c = c[c.stationid==sid]
        if len(c)==1:
            easts.append(float(c.easting))
            norths.append(float(c.northing))

    offset_easting = np.min(easts)
    offset_northing = np.min(norths)
    easts = np.array(easts)
    norths = np.array(norths)
    easts -= offset_easting
    norths -= offset_northing
    for i in range(len(easts)):
        plt.scatter(easts[i], norths[i], color=colors[i], label=k[i], s=100, alpha=.8)
    m=10
    plt.xlim(np.min(easts)-m, np.max(easts)+m)   
    plt.ylim(np.min(norths)-m, np.max(norths)+m)   
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    plt.xlabel("easting distance in cm")
    plt.ylabel("northing distance in cm")
    plt.title("avg GPS position of station "+str(sid))
    return offset_easting, offset_northing

ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')    
def lla2ecef(lat,lon,alt, isradians=True):
    return pyproj.transform(lla, ecef, lon, lat, alt, radians=isradians)

def ecef2lla(X,Y,Z, isradians=True):
    lon, lat, alt = pyproj.transform(ecef, lla, X,Y,Z, radians=isradians)
    return lat, lon, alt



from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def train_test_split(c, train_size=.7):
#    ctrain = c.sample(int(len(c)*train_size))
    c = c.iloc[np.argsort(c.gpstime)]
    ctrain = c.iloc[:int(len(c)*train_size)]
    ctrain = ctrain.iloc[np.argsort(ctrain.gpstime)]
    ctest = pd.DataFrame([i[1] for i in c.iterrows() if not i[0] in ctrain.index])
    ctest = ctest.iloc[np.argsort(ctest.gpstime)]
    return ctrain, ctest
    

def build_NEH_model(c, input_columns, predict_column,  train_days, test_days, estimator, polynomial_expand, verbose=True):
    from sklearn.decomposition import PCA
    ctrain = extract_days(c, train_days)    
    ctest  = extract_days(c, test_days)    
    
    import sys
    Xtrain = ctrain[input_columns]
    ytrain = ctrain[predict_column]
    Xtest  = ctest[input_columns]
    ytest  = ctest[predict_column]

    
    pca = None
    if polynomial_expand is not None:
        Xtrain = polynomial_expand.fit_transform(Xtrain)
        Xtest  = polynomial_expand.transform(Xtest)
    
    if verbose:
        print "TRAIN DATA:", Xtrain.shape, "data points", "    TEST DATA:", Xtest.shape, "data points"
        print "      from:", gpssecs_to_utc(np.min(ctrain.gpstime)), "      from:", gpssecs_to_utc(np.min(ctest.gpstime))
        print "        to:", gpssecs_to_utc(np.max(ctrain.gpstime)), "        to:", gpssecs_to_utc(np.max(ctest.gpstime))
            
    rf = estimator
    rf.fit(Xtrain,ytrain);
    
    ytrain_predict = smooth(rf.predict(Xtrain), window_size=10, crop=False)
    ytest_predict = rf.predict(Xtest)
    ytest_predict = smooth(ytest_predict, window_size=20, crop=False)
    
    ytrain_error = 100*np.mean(np.abs(ytrain-ytrain_predict)/ytrain)   
    ytest_error  = 100*np.mean(np.abs(ytest-ytest_predict)/ytest)
    if verbose:
        print "train prediction error %.2f"%ytrain_error+"%"
        print " test prediction error %.2f"%ytest_error+"%"
    return ytrain_error, ytest_error, ctrain, ctest


def measure_single_NEH_model(c, input_columns, predict_column,  train_days, test_days, estimator, polynomial_expand, verbose=True):
    from sklearn.decomposition import PCA
    ctrain = extract_days(c, train_days)    
    ctest  = extract_days(c, test_days)    
    
    import sys
    Xtrain = ctrain[input_columns]
    ytrain = ctrain[predict_column]
    Xtest  = ctest[input_columns]
    ytest  = ctest[predict_column]

    
    pca = None
    if polynomial_expand is not None:
        Xtrain = polynomial_expand.fit_transform(Xtrain)
        Xtest  = polynomial_expand.transform(Xtest)
    
    rf = estimator
    rf.fit(Xtrain,ytrain);
    
    ytrain_predict = smooth(rf.predict(Xtrain), window_size=10, crop=False)
    ytest_predict = rf.predict(Xtest)
    ytest_predict = smooth(ytest_predict, window_size=20, crop=False)
    
    ytrain_error = 100*np.mean(np.abs(ytrain-ytrain_predict)/ytrain)   
    ytest_error  = 100*np.mean(np.abs(ytest-ytest_predict)/ytest)
    
    return ytrain_error, ytest_error

def pct_score(estimator, X, y):
    p = estimator.predict(X)
    score = 100*np.mean(np.abs(y-p)/y)  
    return score

def show_pred(estimator, c, input_columns, predict_column, polynomial_expand=None, title="", dofig=True):
    if dofig:
        plt.figure(figsize=(15,2))
    data = c[input_columns] if polynomial_expand is None else polynomial_expand.transform(c[input_columns])
    y_predict = estimator.predict(data)    
    y_predict = smooth(y_predict, window_size=20, crop=False)    

    plt.scatter(c.gpstime, c[predict_column], color="orange", s=1, label="measured")
    plt.scatter(c.gpstime, y_predict, color="red", s=1, label="predicted")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    plt.title(title)
    plt.xlim(np.min(c.gpstime), np.max(c.gpstime))
    plt.ylabel("max VTEC")
    plt.xlabel("gpstime")
    
def add_second_in_orbit(c, orbit_period = 60*60*24-240 ):
    c["secinorbit"] = c["gpstime"] % orbit_period
    k=60
    c["b_secinorbit"] = (c.secinorbit/k).as_matrix().astype(int)*k
    return c

def add_second_in_day(c, delta=0):
    c["secinday"] = (c["gpstime"]-delta) % (60*60*24)
    k=60
    c["b_secinday"] = (c.secinday/k).as_matrix().astype(int)*k
    return c

def add_gpsday(c):
    c["gpsday"] = [int(gpssecs_to_gpsday(i)) for i in c.gpstime]
    return c

def join_dicts(d1, d2):
    r={}
    for i in set(d1.keys()).union(set(d2.keys())):
        if i in d1.keys() and i in d2.keys():
            r[i] = pd.concat([d1[i], d2[i]])
        elif i in d1.keys():
            r[i] = d1[i]
        elif i in d2.keys():
            r[i] = d2[i]
    return r


def period_compare(c, field, period_field):
    t=0
    cols = np.array([name for name,_ in matplotlib.colors.cnames.iteritems()])
    cols = list(cols[np.random.permutation(len(cols))[:len(np.unique(c.gpsday))]])
    plt.figure(figsize=(15,2))
    for i in np.unique(c.gpsday):
        cd = c[c.gpsday==i]
        plt.scatter(cd[period_field], cd[field], color=cols[t], s=2, alpha=0.3, label="gps day "+str(int(i)))
        t+=1
    plt.xlim(0, 60*60*24)
    plt.title(field)
    
def get_period_stats(c, period_field, fields=["s_height", "s_northing", "s_easting"]):
    r = []
    for i in np.unique(c[period_field]):
        cs = c[c[period_field]==i]
        one = [i]
        for f in fields:
            one.append(np.std(cs[f]))
        r.append(one)
    r = pd.DataFrame(np.array(r), columns=[period_field]+["std_"+i for i in fields])
    return r
    
def plots_periodic(c, field="b_secinorbit", plot_height=True, plot_northing=True, plot_easting=True):
    if plot_height:
        period_compare(c, "s_height", field)
    if plot_northing:
        period_compare(c, "s_northing", field)
    if plot_easting:
        period_compare(c, "s_easting", field)
    plt.xlabel("orbit second")
    
def plot_stds(c, field="b_secinorbit", fields=["s_height", "s_northing", "s_easting"]):
    r = get_period_stats(c, field, fields)
    plt.figure(figsize=(15,2))
    cols = ["red", "blue", "orange", "black", "green", "cyan"]
    i=0
    for f in fields:
        plt.scatter(r[field], r["std_"+f], color=cols[i], s=2, label=f)
        print "mean",f, np.mean(r["std_"+f])
        i+=1
    plt.plot([np.min(c[field]), np.max(c[field])], [0,0], color="black")
    plt.xlim(np.min(c[field]), np.max(c[field]))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    plt.title("standard deviation for all days in each "+field)
    
    
def orbit_period_compare(c, field):
    cols = np.array([name for name,_ in matplotlib.colors.cnames.iteritems()])
    cols = list(cols[np.random.permutation(len(cols))[:len(np.unique(c.gpsday))]])
    t=0
    plt.figure(figsize=(15,2))
    for i in np.unique(c.gpsday):
        cd = c[c.gpsday==i]
        plt.scatter(cd.secinorbit, cd[field], s=2, alpha=0.5, color=cols[t], label="gps day "+str(int(i)))
        t+=1
    plt.xlim(0, 60*60*24)
    plt.title(field)
#    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));

def extract_days(cd, days):
    return cd[extract_days_idxs(cd, days)]

def extract_days_idxs(cd, days):
    ginit, gend = np.min(cd.gpstime), np.max(cd.gpstime)
    idx = np.zeros(len(cd)).astype(int)
    for day in days:
        start = ginit+86400*day
        end   = np.min([start+86400, gend])
        idx = idx | ((cd.gpstime>=start) &(cd.gpstime<end))
    return idx


def latlon_to_pixels(lat, lon, lat_limits, lon_limits, size_x, size_y):
    minlon, maxlon = lon_limits
    minlat, maxlat = lat_limits
    
    px = int(size_x*(lon-minlon)*1./(maxlon-minlon))
    py = size_y - int(size_y*(lat-minlat)*1./(maxlat-minlat))
    return px, py


import math
MERCATOR_RANGE = 256
#from __future__ import division

def  bound(value, opt_min, opt_max):
  if (opt_min != None): 
    value = max(value, opt_min)
  if (opt_max != None): 
    value = min(value, opt_max)
  return value


def  degreesToRadians(deg) :
  return deg * (math.pi / 180.)


def  radiansToDegrees(rad) :
  return rad / (math.pi / 180.)


class G_Point :
    def __init__(self,x=0, y=0):
        self.x = x
        self.y = y



class G_LatLng :
    def __init__(self,lt, ln):
        self.lat = lt
        self.lng = ln


class MercatorProjection :

    def __init__(self) :
      self.pixelOrigin_ =  G_Point( MERCATOR_RANGE / 2., MERCATOR_RANGE / 2.)
      self.pixelsPerLonDegree_ = MERCATOR_RANGE / 360.
      self.pixelsPerLonRadian_ = MERCATOR_RANGE / (2. * math.pi)


    def fromLatLngToPoint(self, latLng, opt_point=None) :
      point = opt_point if opt_point is not None else G_Point(0.,0.)
      origin = self.pixelOrigin_
      point.x = origin.x + latLng.lng * self.pixelsPerLonDegree_
      # NOTE(appleton): Truncating to 0.9999 effectively limits latitude to
      # 89.189.  This is about a third of a tile past the edge of the world tile.
      siny = bound(math.sin(degreesToRadians(latLng.lat)), -0.9999, 0.9999)
      point.y = origin.y + 0.5 * math.log((1 + siny) / (1 - siny)) * -     self.pixelsPerLonRadian_
      return point


    def fromPointToLatLng(self,point) :
          origin = self.pixelOrigin_
          lng = (point.x - origin.x) / self.pixelsPerLonDegree_
          latRadians = (point.y - origin.y) / -self.pixelsPerLonRadian_
          lat = radiansToDegrees(2 * math.atan(math.exp(latRadians)) - math.pi / 2.)
          return G_LatLng(lat, lng)

    #pixelCoordinate = worldCoordinate * pow(2,zoomLevel)

def getCorners(center, zoom, mapWidth, mapHeight):
    scale = 2**zoom
    proj = MercatorProjection()
    centerPx = proj.fromLatLngToPoint(center)
    SWPoint = G_Point(centerPx.x-(mapWidth/2.)/scale, centerPx.y+(mapHeight/2.)/scale)
    SWLatLon = proj.fromPointToLatLng(SWPoint)
    NEPoint = G_Point(centerPx.x+(mapWidth/2.)/scale, centerPx.y-(mapHeight/2.)/scale)
    NELatLon = proj.fromPointToLatLng(NEPoint)
    return {
        'N' : NELatLon.lat,
        'E' : NELatLon.lng,
        'S' : SWLatLon.lat,
        'W' : SWLatLon.lng,
    }


def plot_inmap(centerLat, centerLon, lats, lons, zoom):

    size_x, size_y = 640,640
    centerPoint = G_LatLng(centerLat, centerLon)
    corners = getCorners(centerPoint, zoom, size_x, size_y)
    lat_limits = (corners["S"], corners["N"])
    lon_limits = (corners["W"], corners["E"])

    dmap = CenterMap(lat=centerLat,lon=centerLon,zoom=zoom, size_x=size_x, size_y=size_y, maptype='roadmap', key='AIzaSyAkFlfbOz3ZvrVPY8us8yFsNULm5zC9xfQ',)
    url = dmap.generate_url()

    urllib.urlretrieve(url, "/tmp/aa.jpg")
    img = io.imread("/tmp/aa.jpg")
    plt.imshow(img)


    pixels = np.array([latlon_to_pixels(lats[i], lons[i], lat_limits, lon_limits, size_x, size_y) for i in range(len(lats))])
    pixels = pixels[(pixels[:,0]>0) & (pixels[:,0]<size_x) & (pixels[:,1]>0) & (pixels[:,1]<size_y) ]
    plt.scatter(pixels[:,0],pixels[:,1], s=5, color="gray", alpha=0.2, label="Ionospheric pierce points")


    px, py = latlon_to_pixels(centerLat, centerLon, lat_limits, lon_limits, size_x, size_y)
    plt.scatter(px,py, s=300, alpha=0.8, color="blue", label="Auger Site")
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks([])
    plt.yticks([]);
    return lat_limits, lon_limits, size_x, size_y

    


from sklearn.grid_search import GridSearchCV
class GPSDaysFold:
    def __init__(self, data, n_train_days, n_test_days):
        self.data = data
        self.n_train_days = n_train_days
        self.n_test_days = n_test_days
        self.day_init, self.day_end = int(np.min(data.gpstime)/86400), int(np.max(data.gpstime)/86400)

    def __iter__ (self):
        for i in range(self.day_end - self.day_init - (self.n_train_days + self.n_test_days)):
            days_for_train = range(i,i+self.n_train_days)
            days_for_test  = range(i+self.n_train_days, i+self.n_train_days+self.n_test_days)
            idxs_for_train = extract_days_idxs(dataset, days_for_train)
            idxs_for_test  = extract_days_idxs(dataset, days_for_test)
            yield np.array(idxs_for_train), np.array(idxs_for_test)
            
    def __len__ (self):
            return self.day_end - self.day_init - (self.n_train_days + self.n_test_days)


from sklearn.decomposition import PCA

class polynomial_expand:
    def __init__ (self, pca_components=None, powers = [], roots = [], no_pca=False):
        self.pca_components=pca_components
        self.powers = powers
        self.roots  = roots
        self.no_pca = no_pca
        
    def combine(self, X):
        if type(X)==pd.DataFrame:
            X      = X.as_matrix()
        baseX  = np.abs(X)

        ncols = X.shape[1]
        for i in self.powers:
            X = np.hstack((X, X[:,:ncols]**i))
            
        for i in self.roots:
            X = np.hstack((X, baseX[:,:ncols]**i))
            
        return X
    
    def transform(self, X):
        X = self.ss1.transform(X)
        X = self.combine(X)
        X = self.ss2.transform(X)
        if not self.no_pca:
            X = self.pca.transform(X)        
        return X    
    
    def fit_transform(self, X):
        self.ss1 = StandardScaler()
        self.ss2 = StandardScaler()

        X = pd.DataFrame(self.ss1.fit_transform(X), columns=X.columns)
        X = self.combine(X)
        X = self.ss2.fit_transform(X)
        if self.pca_components is None:
            self.pca_components = X.shape[1]
        
        if not self.no_pca:
            self.pca = PCA(n_components=self.pca_components)
            self.pca.fit(X)
            X = self.pca.transform(X)
        
        return X


def get_cv_score(data, estimator, n_train_days, n_test_days, input_columns, predict_column, 
                 ft=None, transform_all=True, n_jobs=1):
    print ".",
    gf = GPSDaysFold(data, n_train_days, n_test_days)
    X = data[input_columns]#.as_matrix()
    y = data[predict_column]#.as_matrix()
    if ft is not None and transform_all:
        X = ft.fit_transform(X)
    scores = []
    global sc
    def sc(idxs, X, y, estimator, ft):
        tr, ts = idxs
        Xtrain, Xtest = X[tr], X[ts]
        ytrain, ytest = y[tr], y[ts]
        if not ft is None and not transform_all:
            Xtrain = ft.fit_transform(Xtrain)
            Xtest  = ft.transform(Xtest)
        estimator.fit(Xtrain, ytrain)
        score = pct_score(estimator, Xtest, ytest)
        return score
    scores = Parallel(n_jobs=n_jobs)(delayed(sc)(i, X, y, estimator, ft) for i in gf)
    return np.mean(scores), np.std(scores)

