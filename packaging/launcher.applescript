-- Лаунчер Upscaler 16K: запускает launch.sh из Resources/app (поддерживает перетаскивание изображения на иконку)
on run
	my launchApp("")
end run

on open theFiles
	my launchApp(POSIX path of (item 1 of theFiles))
end open

on launchApp(f)
	set base to POSIX path of (path to me)
	set cmd to "/bin/bash " & quoted form of (base & "Contents/Resources/app/launch.sh")
	if f is not "" then set cmd to cmd & " " & quoted form of f
	do shell script cmd & " >/dev/null 2>&1 &"
end launchApp
