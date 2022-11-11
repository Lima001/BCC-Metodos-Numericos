/* Interpolação linear */
inline double interp_lin(double xl, double x0, double x1, double y0, double y1){
    double u = (xl-x0)/(x1-x0);
    return ((1-u)*y0 + u*y1);
}

/* Interpolação bilinear */
inline double interp_bilin(double xl, double yl, double x0, double x1, double y0, double y1, double fx0y0, double fx0y1, double fx1y0, double fx1y1){
    double r1 = interp_lin(xl, x0, x1, fx0y0, fx1y0);
    double r2 = interp_lin(xl, x0, x1, fx0y1, fx1y1);

    return interp_lin(yl, y0, y1, r1, r2);
}