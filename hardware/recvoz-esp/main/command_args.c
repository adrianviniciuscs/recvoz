#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "command_args.h"

static bool number(const char *s, double lo, double hi, bool integer)
{
    if (!s || !*s) return false;
    char *end;
    errno = 0;
    double v = strtod(s, &end);
    return !errno && *end == '\0' && isfinite(v) && v >= lo && v <= hi
        && (!integer || (v == trunc(v) && strspn(s, "0123456789") == strlen(s)));
}
bool command_args_valid(int argc, char **argv)
{
    if (argc < 1 || !argv) return false;
    const char *cmd = argv[0];
    if (!strcmp(cmd, "enroll")) return argc == 1 || (argc == 2 && *argv[1] && strlen(argv[1]) < 16);
    if (!strcmp(cmd, "list") || !strcmp(cmd, "reset") || !strcmp(cmd, "diag") || !strcmp(cmd, "nscal")) return argc == 1;
    if (!strcmp(cmd, "vad")) return argc == 1 || (argc == 2 && number(argv[1], 1, 32767, true));
    if (!strcmp(cmd, "hold")) return argc == 1 || (argc == 2 && number(argv[1], 0, 60000, true));
    if (!strcmp(cmd, "grab")) return argc == 1 || (argc == 2 && number(argv[1], 1, 3, true));
    if (!strcmp(cmd, "tau")) return argc == 1 || (argc == 2 && number(argv[1], 0, 1, false) && strtod(argv[1], NULL) > 0 && strtod(argv[1], NULL) < 1);
    if (strcmp(cmd, "ns")) return false;
    if (argc == 1) return true;
    if (argc == 2) return !strcmp(argv[1], "on") || !strcmp(argv[1], "off");
    if (argc != 3) return false;
    if (!strcmp(argv[1], "auto")) return !strcmp(argv[2], "on") || !strcmp(argv[2], "off");
    if (!strcmp(argv[1], "floor")) return number(argv[2], 0, .5, false);
    if (!strcmp(argv[1], "alpha")) return number(argv[2], 0, 4, false) && strtod(argv[2], NULL) > 0;
    if (!strcmp(argv[1], "beta")) return number(argv[2], 0, 1, false) && strtod(argv[2], NULL) > 0;
    return false;
}
